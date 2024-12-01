import unittest
from unittest.mock import MagicMock, patch

from magic_pdf.libs.Constants import MODEL_NAME
from magic_pdf.model.model_list import AtomicModel
from magic_pdf.model.sub_modules.model_init2 import (
    AtomModelSingleton, DocLayoutYOLOModelWrapper, Layoutlmv3_PredictorWrapper,
    ModelFactory, ModifiedPaddleOCRWrapper, RapidTableModelWrapper,
    StructTableModelWrapper, TableMasterPaddleModelWrapper,
    UnimernetModelWrapper, YOLOv8MFDModelWrapper)


class TestModelWrappers(unittest.TestCase):
    def test_struct_table_model_wrapper(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = "struct_table_result"
        wrapper = StructTableModelWrapper(
            model_path="path/to/model", max_time=60)
        wrapper.model = mock_model
        result = wrapper.predict("input_data")
        self.assertEqual(result, "struct_table_result")

    def test_table_master_paddle_model_wrapper(self):
        mock_model = MagicMock()
        mock_model.img2html.return_value = "table_master_result"
        wrapper = TableMasterPaddleModelWrapper(
            config={"model_dir": "path/to/model", "device": "cpu"})
        wrapper.model = mock_model
        result = wrapper.predict("input_data")
        self.assertEqual(result, "table_master_result")

    def test_rapid_table_model_wrapper(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = "rapid_table_result"
        wrapper = RapidTableModelWrapper()
        wrapper.model = mock_model
        result = wrapper.predict("input_data")
        self.assertEqual(result, "rapid_table_result")

    def test_yolov8_mfd_model_wrapper(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = "yolov8_mfd_result"
        wrapper = YOLOv8MFDModelWrapper(weight="path/to/weight", device="cpu")
        wrapper.model = mock_model
        result = wrapper.predict("input_data")
        self.assertEqual(result, "yolov8_mfd_result")

    def test_unimernet_model_wrapper(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = "unimernet_result"
        wrapper = UnimernetModelWrapper(
            weight_dir="path/to/weight", cfg_path="path/to/config", device="cpu")
        wrapper.model = mock_model
        result = wrapper.predict("input_data")
        self.assertEqual(result, "unimernet_result")

    def test_layoutlmv3_predictor_wrapper(self):
        mock_model = MagicMock()
        mock_model.return_value = "layoutlmv3_result"
        wrapper = Layoutlmv3_PredictorWrapper(
            weight="path/to/weight", config_file="path/to/config", device="cpu")
        wrapper.model = mock_model
        result = wrapper.predict("input_data")
        self.assertEqual(result, "layoutlmv3_result")

    def test_doclayout_yolo_model_wrapper(self):
        mock_model = MagicMock()
        mock_model.predict.return_value = "doclayout_yolo_result"
        wrapper = DocLayoutYOLOModelWrapper(
            weight="path/to/weight", device="cpu")
        wrapper.model = mock_model
        result = wrapper.predict("input_data")
        self.assertEqual(result, "doclayout_yolo_result")

    def test_modified_paddle_ocr_wrapper(self):
        mock_model = MagicMock()
        mock_model.ocr.return_value = "paddle_ocr_result"
        wrapper = ModifiedPaddleOCRWrapper(
            show_log=False, det_db_box_thresh=0.3, lang="en", use_dilation=True, det_db_unclip_ratio=1.8)
        wrapper.model = mock_model
        result = wrapper.predict("input_data")
        self.assertEqual(result, "paddle_ocr_result")


class TestModelFactory(unittest.TestCase):
    @patch('magic_pdf.model.AtomModel.StructTableModelWrapper')
    def test_create_table_model_struct_eqtable(self, MockStructTableModelWrapper):
        mock_model = MagicMock()
        MockStructTableModelWrapper.return_value = mock_model
        model = ModelFactory.create_model(AtomicModel.Table, table_model_name=MODEL_NAME.STRUCT_EQTABLE,
                                          table_model_path="path/to/model", table_max_time=60, device="cpu")
        self.assertEqual(model, mock_model)

    @patch('magic_pdf.model.AtomModel.TableMasterPaddleModelWrapper')
    def test_create_table_model_table_master(self, MockTableMasterPaddleModelWrapper):
        mock_model = MagicMock()
        MockTableMasterPaddleModelWrapper.return_value = mock_model
        model = ModelFactory.create_model(
            AtomicModel.Table, table_model_name=MODEL_NAME.TABLE_MASTER, table_model_path="path/to/model", device="cpu")
        self.assertEqual(model, mock_model)

    @patch('magic_pdf.model.AtomModel.RapidTableModelWrapper')
    def test_create_table_model_rapid_table(self, MockRapidTableModelWrapper):
        mock_model = MagicMock()
        MockRapidTableModelWrapper.return_value = mock_model
        model = ModelFactory.create_model(
            AtomicModel.Table, table_model_name=MODEL_NAME.RAPID_TABLE)
        self.assertEqual(model, mock_model)

    @patch('magic_pdf.model.AtomModel.YOLOv8MFDModelWrapper')
    def test_create_mfd_model(self, MockYOLOv8MFDModelWrapper):
        mock_model = MagicMock()
        MockYOLOv8MFDModelWrapper.return_value = mock_model
        model = ModelFactory.create_model(
            AtomicModel.MFD, mfd_weights="path/to/weight", device="cpu")
        self.assertEqual(model, mock_model)

    @patch('magic_pdf.model.AtomModel.UnimernetModelWrapper')
    def test_create_mfr_model(self, MockUnimernetModelWrapper):
        mock_model = MagicMock()
        MockUnimernetModelWrapper.return_value = mock_model
        model = ModelFactory.create_model(
            AtomicModel.MFR, mfr_weight_dir="path/to/weight", mfr_cfg_path="path/to/config", device="cpu")
        self.assertEqual(model, mock_model)

    @patch('magic_pdf.model.AtomModel.ModifiedPaddleOCRWrapper')
    def test_create_ocr_model(self, MockModifiedPaddleOCRWrapper):
        mock_model = MagicMock()
        MockModifiedPaddleOCRWrapper.return_value = mock_model
        model = ModelFactory.create_model(
            AtomicModel.OCR, show_log=False, det_db_box_thresh=0.3, lang="en", use_dilation=True, det_db_unclip_ratio=1.8)
        self.assertEqual(model, mock_model)

    @patch('magic_pdf.model.AtomModel.Layoutlmv3_PredictorWrapper')
    def test_create_layout_model_layoutlmv3(self, MockLayoutlmv3PredictorWrapper):
        mock_model = MagicMock()
        MockLayoutlmv3PredictorWrapper.return_value = mock_model
        model = ModelFactory.create_model(AtomicModel.Layout, layout_model_name=MODEL_NAME.LAYOUTLMv3,
                                          layout_weights="path/to/weight", layout_config_file="path/to/config", device="cpu")
        self.assertEqual(model, mock_model)

    @patch('magic_pdf.model.AtomModel.DocLayoutYOLOModelWrapper')
    def test_create_layout_model_doclayout_yolo(self, MockDocLayoutYOLOModelWrapper):
        mock_model = MagicMock()
        MockDocLayoutYOLOModelWrapper.return_value = mock_model
        model = ModelFactory.create_model(
            AtomicModel.Layout, layout_model_name=MODEL_NAME.DocLayout_YOLO, doclayout_yolo_weights="path/to/weight", device="cpu")
        self.assertEqual(model, mock_model)


class TestAtomModelSingleton(unittest.TestCase):
    def test_singleton_instance(self):
        instance1 = AtomModelSingleton()
        instance2 = AtomModelSingleton()
        self.assertIs(instance1, instance2)

    @patch('magic_pdf.model.AtomModel.get_cached_model')
    def test_get_atom_model(self, MockGetCachedModel):
        mock_model = MagicMock()
        MockGetCachedModel.return_value = mock_model
        singleton = AtomModelSingleton()
        model = singleton.get_atom_model(AtomicModel.Table, table_model_name=MODEL_NAME.STRUCT_EQTABLE,
                                         table_model_path="path/to/model", table_max_time=60, device="cpu")
        self.assertEqual(model, mock_model)


if __name__ == '__main__':
    unittest.main()
