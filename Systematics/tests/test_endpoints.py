import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import uproot
from Analysis.simulation_metadata import MUON_SCALE_FACTORS, MUON_SCALE_MODEL
from BackgroundRemoval import common
from Systematics.train import read_sample, train
from omnifold.ratios import RatioConfig, RatioEstimator


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.nominal = self.root / 'nominal.root'
        self.varied = self.root / 'varied.root'
        self.write(self.nominal)
        self.write(self.varied, shower='herwig', factor=2)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, *, shower='pythia', scale='nominal', factor=1,
              process='gg_H', lumi=312., decorated=False, reverse=False):
        count=120
        digest=hashlib.sha256(b'common LHE').hexdigest()
        arrays={name: np.linspace(-.8, .8, count).astype(np.float32) for name in common.FEATURES}
        arrays['reco_m_Z1'] += 90
        arrays['reco_m_Z2'] += 25
        arrays.update(weight=np.full(count, float(factor)), luminosity_fb=np.full(count, lumi),
                      reconstructed=np.ones(count, dtype=bool), reco_m_ZZ=np.full(count, 125.),
                      fiducial=np.ones(count, dtype=bool), lhe_available=np.ones(count, dtype=bool),
                      lhe_source_id=np.full(count, int(digest[:16],16), dtype=np.uint64),
                      lhe_event_index=np.arange(1,count+1,dtype=np.int64))
        arrays['weight'][7]=-0.1*factor
        # Exact boundaries and unselected rows cannot enter training/yields.
        arrays['reco_m_ZZ'][0]=115
        arrays['reco_m_ZZ'][1]=130
        arrays['reconstructed'][2]=False
        if reverse:
            arrays={name: values[::-1] for name,values in arrays.items()}
        with uproot.recreate(path) as output:
            output['Analysis']=arrays
            output['merge_metadata']=json.dumps({'format_version':2,'weight_semantics':'expected_events_at_luminosity'})
            output['lhe_metadata']=json.dumps({'schema_version':1,'sources':[
                {'process':process,'shower':shower,'lhe_sha256':digest,'lhe_source_id':digest[:16]}]})
            output['simulation_metadata']=json.dumps({'schema_version':1,'muon_scale_variation':scale,
                'muon_scale_factor':MUON_SCALE_FACTORS[scale],'muon_scale_one_sigma':.01,
                'muon_scale_model':MUON_SCALE_MODEL,'legacy_nominal_inputs':[]})
            if decorated:
                output['smeft_hb_metadata']='{}'

    def test_signed_weights_selection_and_pair_splits(self):
        a=read_sample(self.nominal,shower='pythia',scale='nominal',region='signal')
        self.write(self.varied,scale='up',reverse=True,factor=2)
        b=read_sample(self.varied,shower='pythia',scale='up',region='signal')
        self.assertEqual(len(a.weights),117)
        self.assertEqual(np.count_nonzero(a.weights<0),1)
        np.testing.assert_array_equal(a.splits,b.splits[::-1])
        self.assertAlmostEqual(b.weights.sum()/a.weights.sum(),2.)
        self.assertEqual(a.features.shape[1],7)

    def test_mislabeled_shower_scale_and_deformation_fail(self):
        for kwargs in ({'shower':'herwig'},{'scale':'up'},{'decorated':True}):
            with self.subTest(kwargs=kwargs):
                self.write(self.nominal,**kwargs)
                with self.assertRaises(ValueError):
                    read_sample(self.nominal,shower='pythia',scale='nominal',region='signal')

    def test_mismatched_luminosity_fails_before_training(self):
        self.write(self.varied,shower='herwig',lumi=139.)
        with patch.object(RatioEstimator,'fit') as fit:
            with self.assertRaisesRegex(ValueError,'luminosity'):
                train(self.nominal,self.varied,variation='theory',output_dir=self.root/'model')
            fit.assert_not_called()

    def test_nonzero_penalty_fails_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError,'must be zero'):
            train(self.root/'missing',self.varied,variation='theory',output_dir=self.root/'model',
                  config=RatioConfig(weight_decay=.01))

    def test_real_nsbi_artifact_retains_density_and_yield(self):
        import torch
        torch.set_num_threads(1)
        config=RatioConfig(hidden_layers=1,neurons=4,batch_size=16,maximum_epochs=1,
                           steps_per_epoch=1,patience=2,max_retries=0,
                           baseline_tolerance=.5,device='cpu')
        model=train(self.nominal,self.varied,variation='theory',output_dir=self.root/'model',
                    config=config,reports=True)
        a=read_sample(self.nominal,shower='pythia',scale='nominal',region='signal')
        prediction=model.predict(a.features)
        self.assertAlmostEqual(model.yield_ratio,2.)
        np.testing.assert_allclose(prediction['full_ratio'],2*prediction['shape_ratio'])
        self.assertEqual(model.manifest['metadata']['variation'],'theory')
        self.assertEqual(model.manifest['architecture']['weight_decay'],0.)
        self.assertEqual(len(model.manifest['members']),4)
        self.assertTrue((self.root/'model/reports/closure.pdf').is_file())
        loaded=RatioEstimator.load(self.root/'model',device='cpu')
        np.testing.assert_allclose(loaded.predict(a.features)['full_ratio'],prediction['full_ratio'])
        with self.assertRaises(FileExistsError):
            train(self.nominal,self.varied,variation='theory',output_dir=self.root/'model',config=config)

if __name__=='__main__':
    unittest.main()
