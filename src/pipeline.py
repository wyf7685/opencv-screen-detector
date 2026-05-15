"""Pipeline模块 - 整合Stage1和Stage2的图像来源识别"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from src.feature.artifact import analyze_artifact
from src.feature.blackscreen import analyze_blackscreen
from src.feature.color_noise import analyze_color_noise
from src.feature.display_content import analyze_display_content
from src.feature.frequency import analyze_frequency
from src.feature.illumination import analyze_illumination
from src.feature.softness import analyze_softness
from src.ml.predict import load_model, predict
from src.preprocess import preprocess_image
from src.stage1.screen_classifier import classify_screen_content
from src.stage1.ui_detector import detect_ui_content
from src.stage2.cfa_artifact import analyze_cfa_artifact
from src.stage2.jpeg_fingerprint import analyze_jpeg_fingerprint
from src.stage2.moire import analyze_moire
from src.stage2.perspective import analyze_perspective
from src.stage2.reflection import analyze_reflection
from src.stage2.rolling_shutter import analyze_rolling_shutter
from src.stage2.sensor_noise import analyze_sensor_noise
from src.stage2.subpixel import analyze_subpixel_fringing


class ImageSourceClassifier:
    """
    图像来源分类器

    二分类：
    - normal: 系统截图和普通照片
    - screen_photo: 手机拍摄屏幕
    """

    def __init__(self, model_path: str | None = None, max_workers: int = 4) -> None:
        """
        初始化分类器

        Args:
            model_path: 模型文件路径（可选）
            max_workers: 并行处理线程数
        """
        self.model = None
        if model_path:
            self.model = load_model(model_path)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __del__(self) -> None:
        self._executor.shutdown(wait=False)

    def classify(self, image: np.ndarray) -> dict:
        """
        分类图像来源

        Args:
            image: BGR格式的输入图像

        Returns:
            分类结果字典
        """
        if image is None or image.size == 0:
            return self._empty_result()

        # Stage1: 屏幕内容检测
        screen_content = self._detect_screen_content(image)

        # Stage2: 成像链特征提取
        imaging_features = self._extract_imaging_features(image)

        # 综合判断
        return self._make_decision(screen_content, imaging_features)

    def _detect_screen_content(self, image: np.ndarray) -> dict[str, float]:
        """
        Stage1: 检测屏幕内容

        Args:
            image: BGR格式图像

        Returns:
            屏幕内容特征
        """
        ui_features = detect_ui_content(image)
        screen_probability = classify_screen_content(ui_features)

        return {
            "ui_features": ui_features,
            "screen_probability": screen_probability,
        }

    def _extract_imaging_features(self, image: np.ndarray) -> dict[str, float]:
        """
        Stage2: 提取成像链特征（并行执行）

        Args:
            image: BGR格式图像

        Returns:
            成像链特征
        """
        processed = preprocess_image(image)

        feature_tasks = {
            # Stage2 imaging chain features
            "perspective": (analyze_perspective, image),
            "sensor_noise": (analyze_sensor_noise, image),
            "cfa_artifact": (analyze_cfa_artifact, image),
            "jpeg_fingerprint": (analyze_jpeg_fingerprint, image),
            "moire": (analyze_moire, image),
            "reflection": (analyze_reflection, image),
            "subpixel_fringing": (analyze_subpixel_fringing, image),
            "rolling_shutter": (analyze_rolling_shutter, image),
            # Old system features for screenshot vs normal_photo distinction
            "frequency": (analyze_frequency, processed),
            "display_content": (analyze_display_content, image),
            "artifact": (analyze_artifact, processed),
            "softness": (analyze_softness, image),
            "blackscreen": (analyze_blackscreen, image),
            "illumination": (analyze_illumination, processed),
            "color_noise": (analyze_color_noise, image),
        }

        features = {}
        future_to_name = {
            self._executor.submit(func, arg): name
            for name, (func, arg) in feature_tasks.items()
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                features[name] = future.result()
            except Exception:
                features[name] = 0.0

        return features

    def _make_decision(
        self,
        screen_content: dict,
        imaging_features: dict[str, float],
    ) -> dict:
        """
        综合判断图像来源

        Args:
            screen_content: 屏幕内容特征
            imaging_features: 成像链特征

        Returns:
            分类结果
        """
        screen_probability = screen_content["screen_probability"]

        # 如果有ML模型，使用模型预测
        if self.model is not None:
            predicted_class, confidence, probabilities = predict(
                self.model, imaging_features
            )

            return {
                "source_class": predicted_class,
                "source_probability": float(confidence),
                "screen_probability": float(screen_probability),
                "features": imaging_features,
                "ui_features": screen_content["ui_features"],
                "class_probabilities": {
                    "screenshot": float(probabilities[0]),
                    "screen_photo": float(probabilities[1]),
                    "normal_photo": float(probabilities[2]),
                },
            }

        # 如果没有模型，使用规则判断
        return self._rule_based_decision(
            screen_probability, imaging_features, screen_content
        )

    def _rule_based_decision(
        self,
        screen_probability: float,
        imaging_features: dict[str, float],
        screen_content: dict,
    ) -> dict:
        """
        基于规则的判断

        Args:
            screen_probability: 屏幕内容概率
            imaging_features: 成像链特征
            screen_content: 屏幕内容特征（包含ui_features）

        Returns:
            分类结果
        """
        # 计算成像链特征得分
        imaging_score = self._calculate_imaging_score(imaging_features)

        # 后处理规则：处理极端sensor_noise情况
        # sensor_noise > 0.95 通常是截图或普通照片的噪声
        # 旧系统Rule 1: sensor>0.95 + softness<0.74 + artifact<0.10
        # 旧系统Rule 13: sensor>0.95 + blackscreen>0.80 + artifact>0.20
        sensor_noise = imaging_features.get("sensor_noise", 0.0)
        if sensor_noise > 0.95:
            # 极端sensor_noise时降低imaging_score，避免误判为screen_photo
            imaging_score = max(0.0, imaging_score - 0.35)
            # 使用text_density作为辅助判断（ui_line_density对所有图像都返回1.0不可靠）
            # text_density > 0.05 说明有文本内容，可能是截图
            if screen_content.get("ui_features", {}).get("text_density", 0.0) > 0.05:
                # 有文本内容，可能是截图 → 强制进入screenshot分支
                screen_probability = 0.30
            else:
                # 无文本内容，可能是普通照片 → 强制进入normal_photo分支
                screen_probability = 0.20

        # 综合判断
        # 使用旧系统的二分类逻辑：normal vs screen_photo
        # 计算综合得分（基于旧系统的特征权重）
        combined_score = self._calculate_combined_score(
            screen_probability, imaging_features
        )

        if combined_score >= 0.23:
            source_class = "screen_photo"
            source_probability = combined_score
        else:
            source_class = "normal"
            source_probability = 1.0 - combined_score

        # 跳过原来的三分类逻辑
        if False:
            # 根据成像链特征判断是截图还是拍屏
            # 截图的成像链特征应该较低（没有摄像头特征）
            # 拍屏的成像链特征应该较高（有摄像头特征）
            if imaging_score > 0.30:
                source_class = "screen_photo"
                source_probability = imaging_score
            else:
                source_class = "screenshot"
                source_probability = 1.0 - imaging_score

        return {
            "source_class": source_class,
            "source_probability": float(source_probability),
            "screen_probability": float(screen_probability),
            "features": imaging_features,
            "imaging_score": float(imaging_score),
        }

    def _calculate_imaging_score(self, features: dict[str, float]) -> float:
        """
        计算成像链特征得分

        Args:
            features: 成像链特征

        Returns:
            成像链得分 (0.0-1.0)
        """
        # 特征权重 - 调整权重以更好地区分截图和拍屏
        # 移除cfa_artifact，因为检测算法对所有图像都返回高值
        # 降低reflection权重，因为普通图片也可能有高光区域
        weights = {
            "perspective": 0.25,
            "sensor_noise": 0.30,
            "cfa_artifact": 0.00,  # 移除权重，因为检测算法有问题
            "jpeg_fingerprint": 0.10,
            "moire": 0.15,
            "reflection": 0.05,  # 降低权重
            "subpixel_fringing": 0.10,
            "rolling_shutter": 0.05,
        }

        # 加权计算
        score = 0.0
        for feature_name, weight in weights.items():
            feature_value = features.get(feature_name, 0.0)
            score += feature_value * weight

        return min(max(score, 0.0), 1.0)

    def _calculate_screenshot_score(self, features: dict[str, float]) -> float:
        """
        计算截图得分（用于区分截图和普通照片）

        基于旧系统的特征权重：
        - 正权重（越高越可能是拍屏/普通照片）：sensor_noise, softness, illumination
        - 负权重（越高越可能是截图）：frequency, display_content, artifact

        Args:
            features: 特征字典

        Returns:
            截图得分 (0.0-1.0)，值越高越可能是截图/普通照片
        """
        weights = {
            "sensor_noise": 0.15,
            "softness": 0.181,
            "illumination": 0.131,
            "frequency": -0.15,
            "display_content": -0.116,
            "artifact": -0.081,
            "blackscreen": 0.03,
        }

        score = 0.0
        for feature_name, weight in weights.items():
            feature_value = features.get(feature_name, 0.0)
            score += feature_value * weight

        return min(max(score, 0.0), 1.0)

    def _calculate_combined_score(
        self,
        screen_probability: float,  # noqa: ARG002
        features: dict[str, float],
    ) -> float:
        """
        计算综合得分（基于旧系统的特征权重）

        旧系统使用加权求和 + 阈值0.23进行二分类：
        - 正权重（越高越可能是screen_photo）：sensor_noise, softness, illumination
        - 负权重（越高越可能是normal）：frequency, display_content, artifact

        Args:
            screen_probability: 屏幕内容概率
            features: 特征字典

        Returns:
            综合得分 (0.0-1.0)
        """
        weights = {
            "sensor_noise": 0.15,
            "softness": 0.181,
            "illumination": 0.131,
            "frequency": -0.15,
            "display_content": -0.116,
            "artifact": -0.081,
            "blackscreen": 0.03,
        }

        score = 0.0
        for feature_name, weight in weights.items():
            feature_value = features.get(feature_name, 0.0)
            score += feature_value * weight

        return min(max(score, 0.0), 1.0)

    def _empty_result(self) -> dict:
        """
        返回空结果

        Returns:
            空结果字典
        """
        return {
            "source_class": "unknown",
            "source_probability": 0.0,
            "screen_probability": 0.0,
            "features": {},
            "error": "Invalid input image",
        }
