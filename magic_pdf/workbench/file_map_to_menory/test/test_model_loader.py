import json
import mmap
import os
import unittest
from tempfile import TemporaryDirectory

from MinerU.magic_pdf.workbench.file_map_to_menory.model_loader import (
    MetaInfoManager, MultiFileModelLoader)


class TestModelLoader(unittest.TestCase):
    def setUp(self):
        # 创建临时目录和文件
        self.temp_dir = TemporaryDirectory()
        self.model_file_paths = [
            os.path.join(
                self.temp_dir.name, 'D:\ProgramData\.cache\hub\opendatalab\PDF-Extract-Kit-1___0\models\MFD\YOLO\yolo_v8_ft.pt'),
            # os.path.join(self.temp_dir.name, 'model_part2.bin')
        ]

        # 创建测试模型文件
        for file_path in self.model_file_paths:
            with open(file_path, 'wb') as f:
                f.write(os.urandom(1024))  # 写入随机数据

        # 创建配置文件
        self.config_path = os.path.join(self.temp_dir.name, 'config.json')
        with open(self.config_path, 'w') as f:
            json.dump({'meta_info_file_path': os.path.join(
                self.temp_dir.name, 'meta_info.json')}, f)

        # 初始化模型加载器
        self.loader = MultiFileModelLoader(
            self.model_file_paths, self.config_path)

    def tearDown(self):
        # 清理临时目录和文件
        self.temp_dir.cleanup()

    def test_load_config(self):
        config = self.loader._load_config()
        self.assertIn('meta_info_file_path', config)
        self.assertEqual(config['meta_info_file_path'], os.path.join(
            self.temp_dir.name, 'meta_info.json'))

    def test_generate_and_save_meta_info(self):
        meta_info = self.loader._generate_meta_info()
        self.loader.save_meta_info(meta_info)
        loaded_meta_info = self.loader.load_meta_info()
        self.assertIsNotNone(loaded_meta_info)
        for file_path in self.model_file_paths:
            self.assertIn(file_path, loaded_meta_info)
            self.assertEqual(
                loaded_meta_info[file_path]['file_size'], os.path.getsize(file_path))
            self.assertEqual(
                loaded_meta_info[file_path]['last_modified'], os.path.getmtime(file_path))

    def test_load_model(self):
        models = self.loader.load_model()
        self.assertIsNotNone(models)
        self.assertEqual(len(models), len(self.model_file_paths))
        for model in models:
            self.assertIsInstance(model, mmap.mmap)
            model.close()

    def test_is_meta_info_outdated(self):
        meta_info = self.loader._generate_meta_info()
        self.assertFalse(self.loader._is_meta_info_outdated(meta_info))

        # 修改一个文件的最后修改时间
        os.utime(self.model_file_paths[0], (os.path.getatime(
            self.model_file_paths[0]), os.path.getmtime(self.model_file_paths[0]) + 1))
        self.assertTrue(self.loader._is_meta_info_outdated(meta_info))


if __name__ == '__main__':
    unittest.main()
