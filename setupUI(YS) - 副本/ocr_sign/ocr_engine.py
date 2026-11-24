"""
OCR路牌识别模块
用于比赛Task 2: 语义导航（识别路牌文字+图标+箭头）

技术路线:
1. 目标检测先定位路牌区域
2. 对路牌区域做OCR提取文字
3. 将文字+图标信息发送给文心大模型API
4. 大模型返回左转/右转导航指令

注意: 比赛要求必须实时调用OCR+大模型API，不能写死规则
"""

import cv2
import numpy as np
import json
import base64
import requests


class OCRSignRecognizer:
    """
    路牌OCR识别器

    用法:
        recognizer = OCRSignRecognizer()
        text, direction = recognizer.recognize(sign_img)
    """

    def __init__(self, ernie_api_key="", ernie_secret_key="", ernie_access_token="", qianfan_api_key=""):
        """
        参数:
            ernie_api_key: 文心大模型API Key (旧版aip API, 已弃用)
            ernie_secret_key: 文心大模型Secret Key (旧版aip API, 已弃用)
            ernie_access_token: AI Studio access_token (erniebot SDK用, 仅文字模型)
            qianfan_api_key: 千帆v2 API Key (bce-v3格式, 调ernie-5.0文字大模型)
        """
        self.api_key = ernie_api_key
        self.secret_key = ernie_secret_key
        self.access_token = ernie_access_token or None
        self.qianfan_api_key = qianfan_api_key or None

        # OCR引擎：优先使用PaddleOCR（飞桨生态）
        self.ocr_engine = None
        self._init_ocr()

    def _init_ocr(self):
        """初始化OCR引擎（优先PaddleOCR，其次Tesseract）"""
        self.ocr_type = None
        # ① 先试PaddleOCR
        try:
            from paddleocr import PaddleOCR
            try:
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            except TypeError:
                self.ocr_engine = PaddleOCR(lang='ch')
            self.ocr_type = "paddle"
            print("[OCR] PaddleOCR 初始化成功")
            return
        except Exception as e:
            print(f"[OCR] PaddleOCR 不可用: {type(e).__name__}")

        # ② 再试Tesseract(免费开源, apt直接装)
        try:
            import pytesseract
            self.ocr_engine = pytesseract
            self.ocr_type = "tesseract"
            print("[OCR] Tesseract OCR 初始化成功")
            return
        except Exception as e:
            print(f"[OCR] Tesseract 不可用: {type(e).__name__}: {e}")

        self.ocr_engine = None
        print("[OCR] 所有OCR引擎均不可用, 路牌将默认right")

    def extract_text(self, img):
        """从图像提取文字(兼容PaddleOCR/Tesseract)"""
        if self.ocr_engine is None:
            return []
        try:
            if self.ocr_type == "tesseract":
                # Tesseract OCR: 中文+英文, psm 7=单行文字
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                text = self.ocr_engine.image_to_string(
                    gray, lang='chi_sim+eng', config='--psm 7')
                texts = [t.strip() for t in text.split('\n') if t.strip()]
                return texts
            else:
                # PaddleOCR: 3.x 用 predict(), 2.x 用 ocr()
                if hasattr(self.ocr_engine, 'predict'):
                    result = self.ocr_engine.predict(img)
                else:
                    try:
                        result = self.ocr_engine.ocr(img, cls=True)
                    except TypeError:
                        result = self.ocr_engine.ocr(img)
                return self._parse_ocr_result(result)
        except Exception as e:
            print(f"[OCR] 识别失败: {type(e).__name__}: {e}")
            return []

    def _parse_ocr_result(self, result):
        """解析OCR结果(兼容2.x list格式 / 3.x dict格式)"""
        texts = []
        try:
            # 3.x dict 格式: rec_texts + rec_scores
            if isinstance(result, dict):
                rec_texts = result.get('rec_texts') or result.get('text') or []
                rec_scores = result.get('rec_scores') or []
                for i, t in enumerate(rec_texts):
                    c = rec_scores[i] if i < len(rec_scores) else 0.9
                    if c > 0.5:
                        texts.append(str(t))
            # 3.x 可能返回 list[dict]
            elif isinstance(result, list) and result and isinstance(result[0], dict):
                for r in result:
                    rec_texts = r.get('rec_texts') or []
                    rec_scores = r.get('rec_scores') or []
                    for i, t in enumerate(rec_texts):
                        c = rec_scores[i] if i < len(rec_scores) else 0.9
                        if c > 0.5:
                            texts.append(str(t))
            # 2.x list 格式: [[box,(text,conf)],...]
            elif isinstance(result, list) and result and isinstance(result[0], list):
                for line in result[0]:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        info = line[1]
                        if isinstance(info, (list, tuple)) and len(info) >= 2:
                            text, conf = info[0], info[1]
                            if conf > 0.5:
                                texts.append(str(text))
        except Exception as e:
            print(f"[OCR] 解析失败: {type(e).__name__}: {e}")
        return texts

    def ask_ernie_direction(self, sign_text):
        """
        调用千帆ernie-5.0大模型，根据路牌文字判断方向（理解反讽）

        参数:
            sign_text: str, 路牌上识别到的文字

        返回:
            direction: str, "left" / "right" / "straight"
            raw_response: str, 大模型原始回复
        """
        # 优先用千帆v2 API + ernie-5.0
        if self.qianfan_api_key:
            prompt = (
                f"你是自动驾驶导航助手。赛道路牌上写着：{sign_text}\n"
                f"请判断应该左转、右转还是直行。\n"
                f"路牌可能使用各种表达方式，要理解真实含义：\n"
                f"- 直接指令: '请右转' → right; '请左转' → left; '直行' → straight\n"
                f"- 反讽否定: '右道比直道好走多了，才怪！' → straight\n"
                f"- 反讽否定: '右侧是一条不归路' → straight\n"
                f"- 间接引导: '左侧道路崎岖，右侧一马平川' → right\n"
                f"- 禁止通行: '禁止直行' → left或right(不能直走)\n"
                f"- 条件推理: '右道真的崎岖，但是直道真的走不了了' → right\n"
                f"只回答一个词：left、right 或 straight"
            )
            url = "https://qianfan.baidubce.com/v2/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.qianfan_api_key}"
            }
            payload = {
                "model": "ernie-5.0",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=5.0)
                result = resp.json()
                if "error" in result:
                    print(f"[ERNIE] 大模型错误: {result.get('error', '')}")
                    return self._fallback_direction(sign_text), ""
                raw = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
                print(f"[ERNIE] 大模型回复: {raw}")
                # 安全解析: 先英文, 再中文否定判断, 防止"不要右转"误判为right
                if "straight" in raw:
                    return "straight", raw
                if "right" in raw:
                    return "right", raw
                if "left" in raw:
                    return "left", raw
                # 中文兜底: 含"不"+"右" → straight(不要右转)
                if "直" in raw or ("不" in raw and "右" in raw):
                    return "straight", raw
                if "右" in raw and "不" not in raw:
                    return "right", raw
                if "左" in raw and "不" not in raw:
                    return "left", raw
                return "straight", raw
            except Exception as e:
                print(f"[ERNIE] API调用失败: {e}")
                return self._fallback_direction(sign_text), ""

        # 没有千帆API Key → 规则匹配
        print("[ERNIE] 未配置千帆API Key，使用规则匹配")
        return self._fallback_direction(sign_text), ""

    def ask_ernie_with_image(self, img):
        """
        路牌图片 → 千帆qianfan-ocr提取文字 → ernie-5.0判断方向(理解反讽)

        两步法:
        1. qianfan-ocr(千帆v2 API)识别路牌图片中的文字
        2. ernie-5.0(千帆v2 API)根据文字判断方向(能理解反讽)

        参数:
            img: 路牌区域的BGR图像
        返回:
            direction: str, "left" / "right" / "straight"
            raw_response: str, 大模型原始回复
        """
        if not self.qianfan_api_key:
            print("[ERNIE-V] 未配置千帆API Key → 默认right")
            return "right", ""

        url = "https://qianfan.baidubce.com/v2/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.qianfan_api_key}"
        }

        # ===== Step 1: deepseek-ocr提取文字 =====
        try:
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            data_url = f"data:image/jpeg;base64,{img_b64}"
            ocr_payload = {
                "model": "qianfan-ocr",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "识别图片中的所有文字，只返回文字内容"}
                    ]
                }]
            }
            resp = requests.post(url, headers=headers, json=ocr_payload, timeout=10.0)
            result = resp.json()
            if "error" in result:
                print(f"[ERNIE-V] OCR错误: {result.get('error', '')}")
                return "right", ""
            sign_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            # 也检查text字段(部分模型返回格式不同)
            if not sign_text:
                sign_text = result.get("text", "").strip()
            print(f"[ERNIE-V] OCR识别: '{sign_text}'")
        except Exception as e:
            print(f"[ERNIE-V] OCR失败: {type(e).__name__}: {e}")
            return "straight", ""

        if not sign_text:
            print("[ERNIE-V] OCR没识别到文字 → 默认straight")
            return "straight", ""
        if len(sign_text) < 5:
            print(f"[ERNIE-V] 文字太短'{sign_text}', 路牌可能没显示全 → straight")
            return "straight", ""

        # ===== Step 2: ernie-5.0判断方向(理解反讽) =====
        prompt = (
            f"你是自动驾驶导航助手。赛道路牌上写着：{sign_text}\n"
            f"请判断应该左转、右转还是直行。\n"
            f"注意路牌可能使用反讽或否定语气，要理解真实含义：\n"
            f"- '右道比直道好走多了，才怪！' → 不要右转 → straight\n"
            f"- '右侧是一条不归路' → 不要右转 → straight\n"
            f"- '左侧道路崎岖，右侧一马平川' → 要右转 → right\n"
            f"- '右道真的崎岖，但是直道真的走不了了' → 要右转 → right\n"
            f"只回答一个英文单词：left、right 或 straight，不要回答任何中文或其他文字"
        )
        llm_payload = {
            "model": "ernie-5.0",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        try:
            resp = requests.post(url, headers=headers, json=llm_payload, timeout=5.0)
            result = resp.json()
            if "error" in result:
                print(f"[ERNIE-V] 大模型错误: {result.get('error', '')}")
                return "right", ""
            raw = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
            print(f"[ERNIE-V] 大模型回复: {raw}")
            # 优先匹配英文单词(大模型应只返回英文)
            if "straight" in raw:
                return "straight", raw
            if "right" in raw:
                return "right", raw
            if "left" in raw:
                return "left", raw
            # 中文兜底: 含"不"+"右"=不右转=straight; 含"直"=straight
            if "直" in raw or ("不" in raw and "右" in raw):
                return "straight", raw
            if "右" in raw and "不" not in raw:
                return "right", raw
            if "左" in raw and "不" not in raw:
                return "left", raw
            # 默认不右转(安全)
            return "straight", raw
        except Exception as e:
            print(f"[ERNIE-V] 大模型调用失败: {type(e).__name__}: {e}")
            return "right", ""

    def _get_access_token(self):
        """获取文心大模型access_token"""
        if self.access_token:
            return self.access_token

        url = (
            "https://aip.baidubce.com/oauth/2.0/token"
            f"?grant_type=client_credentials&client_id={self.api_key}"
            f"&client_secret={self.secret_key}"
        )
        try:
            resp = requests.post(url, timeout=5.0)
            self.access_token = resp.json().get("access_token", "")
        except Exception as e:
            print(f"[ERNIE] 获取token失败: {e}")
        return self.access_token

    def _fallback_direction(self, text):
        """
        降级方案：当无法调用大模型时，用简单规则匹配
        注意：比赛正式时必须用大模型API，此方法仅供调试
        """
        if not text:
            return "straight"
        # 仅调试用
        if "左" in text:
            return "left"
        elif "右" in text:
            return "right"
        return "straight"

    def recognize(self, sign_img):
        """
        完整流程：OCR提取文字 → 文心大模型判断方向

        参数:
            sign_img: 路牌区域的BGR图像

        返回:
            texts: list[str], OCR识别到的文字
            direction: str, 导航方向 "left"/"right"/"straight"
            raw_response: str, 大模型原始回复
        """
        # Step 1: OCR提取文字
        texts = self.extract_text(sign_img)
        sign_text = " ".join(texts)

        if not sign_text:
            return [], "straight", ""

        # Step 2: 调用文心大模型
        direction, raw = self.ask_ernie_direction(sign_text)

        return texts, direction, raw
