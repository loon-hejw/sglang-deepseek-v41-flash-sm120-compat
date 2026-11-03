#!/usr/bin/env python3
"""Exercise the public launcher without Docker, GPUs or model weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / 'examples/serve-compat-tp4.sh'


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model with spaces'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.cache = self.root / 'new cache'
        self.capture = self.root / 'docker-args.json'
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        docker = bin_dir / 'docker'
        docker.write_text('#!' + sys.executable + '\n'
                          'import json, os, pathlib, sys\n'
                          'pathlib.Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n')
        docker.chmod(0o755)
        self.env = {'PATH': str(bin_dir) + os.pathsep + os.defpath,
                    'MODEL_DIR': str(self.model), 'CACHE_DIR': str(self.cache),
                    'CAPTURE': str(self.capture), 'API_KEY': 'synthetic-private-value',
                    'LANG': 'C.UTF-8'}

    def launch(self, *args, **env):
        return subprocess.run(['bash', str(LAUNCHER), *args], env={**self.env, **env},
                              text=True, capture_output=True, timeout=15)

    def captured(self):
        return json.loads(self.capture.read_text())

    def value(self, args, flag):
        return args[args.index(flag) + 1]

    def test_default_profile_matches_qualified_engine(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.captured()
        expected = {'--tp-size':'4', '--ep-size':'2', '--mem-fraction-static':'0.85',
                    '--context-length':'1048576', '--max-running-requests':'16',
                    '--speculative-algorithm':'DSPARK', '--speculative-dspark-block-size':'5',
                    '--speculative-dspark-sps-autotune':'auto', '--hicache-ratio':'2',
                    '--attention-backend':'dsv4', '--moe-runner-backend':'flashinfer_mxfp4',
                    '--fp8-gemm-backend':'flashinfer_cutlass', '--sampling-defaults':'openai',
                    '--reasoning-parser':'deepseek-v41', '--tool-call-parser':'deepseekv41',
                    '--served-model-name':'deepseek-v4.1-flash-1017', '--publish':'18001:8000',
                    '--gpus':'"device=0,1,2,3"', '--restart':'no', '--api-key':self.env['API_KEY']}
        for key, value in expected.items():
            with self.subTest(flag=key): self.assertEqual(self.value(args, key), value)
        self.assertIn('--detach', args)
        self.assertNotIn('--rm', args)
        self.assertNotIn('--memory', args)
        self.assertNotIn('--cpus', args)
        self.assertNotIn('--cpuset-mems', args)
        self.assertEqual(self.value(args, '--entrypoint'), 'sglang')
        self.assertIn(str(self.model.resolve()) + ':/models/deepseek-v41-flash:ro', args)
        self.assertIn(str(self.cache.resolve()) + ':/root/.cache', args)
        self.assertIn('CUDA_VISIBLE_DEVICES=0,1,2,3', args)
        self.assertIn('SGLANG_ENABLE_SM120_MXFP8_SKINNY_GEMM=1', args)
        self.assertIn('SGLANG_DSV41_WO_A_FP8_COPY=1', args)
        self.assertIn('SGLANG_MAX_NEW_TOKENS_LIMIT=393216', args)

    def test_preview_redacts_secret_without_writes(self):
        result = self.launch('--dry-run', API_KEY='secret with spaces $() and quotes')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('secret with', result.stdout + result.stderr)
        self.assertIn('REDACTED', result.stdout)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.capture.exists())

    def test_uuid_devices_and_runtime_overrides(self):
        uuids = ','.join(f'GPU-0000000{i}-0000-0000-0000-000000000000' for i in range(4))
        result = self.launch(GPU_DEVICES=uuids, PORT='18002', SERVED_MODEL_NAME='custom-model',
                             CONTAINER_NAME='custom-container', MAX_RUNNING_REQUESTS='4',
                             ENABLE_HICACHE='0', API_KEY='')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.captured()
        self.assertEqual(self.value(args, '--gpus'), '"device=' + uuids + '"')
        self.assertEqual(self.value(args, '--publish'), '18002:8000')
        self.assertEqual(self.value(args, '--served-model-name'), 'custom-model')
        self.assertEqual(self.value(args, '--name'), 'custom-container')
        self.assertEqual(self.value(args, '--max-running-requests'), '4')
        self.assertNotIn('--enable-hierarchical-cache', args)
        self.assertNotIn('--hicache-ratio', args)
        self.assertNotIn('--api-key', args)

    def test_shared_host_profile_bounds_resources_and_keeps_engine(self):
        result = self.launch(HOST_PROFILE='shared', NUMA_MEMORY_NODE='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.captured()
        expected = {'--memory':'224g', '--memory-swap':'224g', '--cpus':'16',
                    '--cpuset-mems':'1', '--entrypoint':'python3', '--hicache-ratio':'0.1',
                    '--tp-size':'4', '--ep-size':'2', '--mem-fraction-static':'0.85',
                    '--max-running-requests':'16'}
        for key, value in expected.items():
            with self.subTest(flag=key): self.assertEqual(self.value(args, key), value)
        self.assertNotIn('--attn-dp-size', args)
        self.assertNotIn('--enable-dp-lm-head', args)
        wrapper_index = args.index('-c')
        self.assertIn('p(41,1,0,0,0)', args[wrapper_index + 1])
        self.assertIn('p(42,0,0,0,0)', args[wrapper_index + 1])
        self.assertEqual(args[wrapper_index + 2:wrapper_index + 4], ['sglang', 'serve'])

    def test_shared_preview_has_no_side_effects_and_redacts_key(self):
        result = self.launch('--dry-run', HOST_PROFILE='shared', NUMA_MEMORY_NODE='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--memory 224g', result.stdout)
        self.assertNotIn(self.env['API_KEY'], result.stdout + result.stderr)
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.capture.exists())

    def test_shared_host_requires_explicit_valid_numa_placement(self):
        for values in [{'HOST_PROFILE':'shared'}, {'HOST_PROFILE':'unknown'},
                       {'NUMA_MEMORY_NODE':'1'}, {'HOST_PROFILE':'shared', 'NUMA_MEMORY_NODE':'0,1'},
                       {'HOST_PROFILE':'shared', 'NUMA_MEMORY_NODE':'-1'},
                       {'HOST_PROFILE':'shared', 'NUMA_MEMORY_NODE':'00'}]:
            with self.subTest(values=values):
                self.assertNotEqual(self.launch(**values).returncode, 0)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.cache.exists())

    def test_fractional_cache_ratio_and_invalid_values(self):
        result = self.launch(HICACHE_RATIO='0.1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.value(self.captured(), '--hicache-ratio'), '0.1')
        self.capture.unlink()
        for value in ['0.0', '-0.1', 'NaN', '1e2', '0.1; echo bad']:
            with self.subTest(value=value):
                self.assertNotEqual(self.launch(HICACHE_RATIO=value).returncode, 0)
                self.assertFalse(self.capture.exists())

    def test_invalid_resource_selection_fails_before_docker(self):
        for devices in ['all', '0,1,2', '0,1,2,3,4', '0,1,1,3', '0,1,2,3,',
                        ',0,1,2', '0,,2,3', '0,1,2,03', '0,1,2,not-a-gpu']:
            with self.subTest(devices=devices):
                self.assertNotEqual(self.launch(GPU_DEVICES=devices).returncode, 0)
                self.assertFalse(self.capture.exists())
                self.assertFalse(self.cache.exists())

    def test_invalid_profile_limits_fail_before_docker(self):
        for values in [{'TP_SIZE':'8'}, {'MAX_RUNNING_REQUESTS':'17'}, {'CONTEXT_LENGTH':'1048577'},
                       {'PORT':'65536'}, {'MEM_FRACTION':'1'}, {'MEM_FRACTION':'0.0'},
                       {'PORT':'08000'}, {'ENABLE_HICACHE':'invalid'}, {'HICACHE_RATIO':'0'}]:
            with self.subTest(values=values):
                self.assertNotEqual(self.launch(**values).returncode, 0)
                self.assertFalse(self.capture.exists())

    def test_missing_model_and_invalid_cli_fail_without_docker(self):
        self.assertNotEqual(self.launch(MODEL_DIR=str(self.root / 'missing')).returncode, 0)
        self.assertNotEqual(self.launch('--not-a-valid-option').returncode, 0)
        self.assertFalse(self.capture.exists())


if __name__ == '__main__':
    unittest.main()
