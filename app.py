import io
import os
import re
import zipfile
import easyocr
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import streamlit as st
from PIL import Image

# 웹 페이지 기본 설정
st.set_page_config(
    page_title="인수검사서 파일명 자동 생성기 | (주)정우",
    page_icon="📄",
    layout="wide",
)

# ---------------- 사이드바 (로고 및 담당자 문의 정보) ----------------
with st.sidebar:
  # 로고 이미지 표시
  logo_path = "logo.jpg"
  if not os.path.exists(logo_path):
    logo_path = "세로-영문-Jeongwoo.jpg"

  if os.path.exists(logo_path):
    st.image(logo_path, use_container_width=True)

  st.markdown("---")
  st.markdown("### 📞 시스템 문의 및 지원")
  st.info("""
    **시스템 개발 및 관리자**
    * **부서:** PS품질팀
    * **담당자:** 김선웅
    * **문의 내용:** 프로그램 오류, 수주번호/업체명 추출 규칙 수정 및 변경 요청
    """)
  st.markdown("---")
  st.caption("ⓒ Jeongwoo Industrial Machine Co., Ltd. All rights reserved.")

# ---------------- 메인 화면 제목 및 업데이트 표시 ----------------
UPDATE_DATE = "2026-09-22"
st.title(f"📄 인수검사서 파일명 자동 생성기 `v{UPDATE_DATE}`")
st.caption(
    f"📅 최종 업데이트: {UPDATE_DATE} | [H/X/Z/M + 숫자] 정밀 수주번호 추출 및 Material 오인식"
    " 방지 적용"
)
st.markdown("---")


# EasyOCR 모델 리더 로드 (CPU 최적화)
@st.cache_resource
def load_ocr_reader():
  return easyocr.Reader(["ko", "en"], gpu=False)


with st.spinner("OCR AI 모델을 로딩 중입니다..."):
  reader = load_ocr_reader()


# 수주번호 오인식 문자 정밀 보정 함수 (O->0, Z->2 등)
def clean_and_fix_order_no(order_str):
  if not order_str:
    return ""

  if any(
      bad in order_str.upper() for bad in ["MATERIAL", "MATER1AL", "MATL"]
  ):
    return ""

  match = re.search(
      r"([HXZM][0-9OQZILsSbB]{5,}(?:[\-\_]?[A-Za-z0-9]+)?)",
      order_str,
      re.IGNORECASE,
  )
  if not match:
    return ""

  raw_order = match.group(1)
  prefix = raw_order[0].upper()
  rest = raw_order[1:]

  parts = re.split(r"([\-_])", rest)
  main_part = parts[0]

  corrected_main = ""
  for idx, char in enumerate(main_part):
    if idx < 6:
      c_upper = char.upper()
      if c_upper in ["O", "Q"]:
        corrected_main += "0"
      elif c_upper == "Z":
        corrected_main += "2"
      elif c_upper in ["I", "L"]:
        corrected_main += "1"
      elif c_upper == "S":
        corrected_main += "5"
      elif c_upper == "B":
        corrected_main += "8"
      else:
        corrected_main += char
    else:
      corrected_main += char

  parts[0] = corrected_main
  return prefix + "".join(parts)


# 스마트 OCR 처리 함수
def process_ocr_smart(img, ocr_reader):
  angles = [0, 270, 90, 180]
  for angle in angles:
    test_img = img.rotate(angle, expand=True) if angle != 0 else img
    w, h = test_img.size

    crop_box = (0, 0, w, int(h * 0.50))
    cropped = test_img.crop(crop_box)
    img_np = np.array(cropped.convert("RGB"))

    results = ocr_reader.readtext(img_np)

    extracted_text = "".join([t[1].strip() for t in results])
    if len(extracted_text) >= 10:
      return results, test_img

  return ocr_reader.readtext(np.array(img.convert("RGB"))), img


# ---------------- 파일 업로드 ----------------
uploaded_files = st.file_uploader(
    "PDF 또는 이미지 파일을 선택하세요 (최대 5개)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
  if len(uploaded_files) > 5:
    st.warning(
        "⚠️ 최대 5개까지 한 번에 처리 가능합니다. 상위 5개 파일만 분석합니다."
    )
    target_files = uploaded_files[:5]
  else:
    target_files = uploaded_files

  processed_results = []
  zip_buffer = io.BytesIO()

  with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
    progress_bar = st.progress(0)

    for idx, file in enumerate(target_files):
      file_bytes = file.read()
      file_ext = file.name.split(".")[-1].lower()

      try:
        image = None
        if file_ext == "pdf":
          pdf = pdfium.PdfDocument(file_bytes)
          page = pdf[0]
          image = page.render(scale=1.8).to_pil()
        else:
          image = Image.open(io.BytesIO(file_bytes))

        if image:
          ocr_results, _ = process_ocr_smart(image, reader)

          parsed_data = []
          full_text_list = []

          for bbox, text, prob in ocr_results:
            text_clean = text.strip()
            if text_clean:
              full_text_list.append(text_clean)
              top_y = bbox[0][1]
              left_x = bbox[0][0]
              parsed_data.append({
                  "text": text_clean,
                  "top": top_y,
                  "left": left_x,
                  "prob": prob,
              })

          full_text = " ".join(full_text_list)
          df = pd.DataFrame(parsed_data) if parsed_data else pd.DataFrame()

        # ---------------- 1. 수주번호 추출 ----------------
        order_no = ""
        order_blacklist = [
            "ORDER",
            "URDER",
            "NUMBER",
            "DELIVER",
            "CUSTOMER",
            "VENDOR",
            "INSPECTION",
            "REPORT",
            "NOTIFICATION",
            "ORDERNO",
            "URDERNO",
            "MATERIAL",
            "MATER1AL",
            "MATL",
        ]

        h_matches = re.findall(
            r"\b([HXZM][0-9OQZILsSbB]{5,}(?:[\-\_]?[A-Za-z0-9]+)?)\b",
            full_text,
            re.IGNORECASE,
        )
        for hm in h_matches:
          hm_upper = hm.upper()
          if hm_upper not in order_blacklist and not hm_upper.startswith(
              "MATER"
          ):
            order_no = hm.strip()
            break

        if not order_no:
          alt_order = re.search(
              r"수주번호(?:[^\w]|Order|Urder|No)*([HXZM][0-9OQZILsSbB]{5,}[A-Za-z0-9\-_]*)",
              full_text,
              re.IGNORECASE,
          )
          if alt_order:
            cand = alt_order.group(1).strip()
            if cand.upper() not in order_blacklist and not cand.upper().startswith(
                "MATER"
            ):
              order_no = cand

        order_no = re.sub(r"[\-_]$", "", order_no)
        order_no = clean_and_fix_order_no(order_no)

        # ---------------- 2. 의뢰일자 추출 ----------------
        date = ""
        date_matches = re.findall(
            r"(20[2-9][0-9][-/.][0-9]{2][-/.][0-9]{2})", full_text
        )
        if date_matches:
          for m in date_matches:
            digits = re.sub(r"[^0-9]", "", m)
            if len(digits) == 8 and digits.startswith("20"):
              date = digits
              break
        if not date:
          for txt in full_text_list:
            digits = re.sub(r"[^0-9]", "", txt)
            if len(digits) == 8 and digits.startswith("20"):
              date = digits
              break

        # ---------------- 3. 업체명 추출 ----------------
        vendor = ""
        customer_words = set()
        if not df.empty:
          cust_labels = df[
              df["text"].str.contains("고객|Customer", na=False, case=False)
          ]
          if not cust_labels.empty:
            c_top, c_left = (
                cust_labels.iloc[0]["top"],
                cust_labels.iloc[0]["left"],
            )
            cust_targets = df[
                (df["top"] >= c_top - 20)
                & (df["top"] <= c_top + 30)
                & (df["left"] > c_left)
            ].sort_values(by="left")
            for _, r in cust_targets.iterrows():
              clean_c = re.sub(r"[^가-힣a-zA-Z0-9]", "", r["text"])
              if clean_c and clean_c not in ["고객", "Customer"]:
                customer_words.add(clean_c)

        if not df.empty:
          vendor_labels = df[
              df["text"].str.contains(
                  "업체소재지|소재지|Vendor", na=False, case=False
              )
          ]
          if not vendor_labels.empty:
            v_row = vendor_labels.iloc[0]
            v_top, v_left = v_row["top"], v_row["left"]
            targets = df[
                (df["left"] > v_left + 5)
                & (df["top"] >= v_top - 40)
                & (df["top"] <= v_top + 50)
            ].sort_values(by="left")
            for _, r in targets.iterrows():
              t = r["text"]
              if any(
                  k in t
                  for k in [
                      "업체소재지",
                      "Vendor",
                      "Address",
                      "결재",
                      "성산구",
                      "의창구",
                      "강서구",
                      "녹산산업",
                  ]
              ):
                continue
              clean_t = re.split(
                  r"[\(\[\d]|경상남도|창원시|의창구|부산|강서구|녹산|경남|서울|경기|시|구|군",
                  t,
              )[0].strip()
              clean_t = re.sub(r"[^가-힣a-zA-Z0-9]", "", clean_t)
              if len(clean_t) >= 2 and clean_t not in customer_words:
                vendor = clean_t
                break

        if not vendor:
          for txt in full_text_list:
            clean_txt = re.split(
                r"[\(\[\d]|경상남도|창원시|의창구|부산|강서구|녹산|경남|서울|경기|시|구|군",
                txt,
            )[0].strip()
            clean_txt = re.sub(r"[^가-힣a-zA-Z0-9]", "", clean_txt)
            if clean_txt in customer_words or clean_txt in [
                "업체소재지",
                "Vendor",
                "Address",
                "고객",
                "Customer",
            ]:
              continue
            if len(clean_txt) >= 2 and any(
                k in txt
                for k in [
                    "스틸",
                    "볼텍",
                    "머티리얼",
                    "에스앤피",
                    "주식회사",
                    "(주)",
                    "공업",
                    "금속",
                    "테크",
                    "산업",
                    "엔지니어링",
                    "상사",
                    "정밀",
                    "기업",
                    "파이프",
                    "금동",
                    "스틱",
                ]
            ):
              vendor = clean_txt
              break

        if vendor:
          vendor = re.sub(r"스틱$", "스틸", vendor)
          vendor = vendor.replace("스틱", "스틸")

        # ---------------- 4. 발주서번호 추출 ----------------
        po_no = ""
        po_match = re.search(r"(PO?[0-9]{8,})", full_text, re.IGNORECASE)
        if po_match:
          po_no = po_match.group(1).strip()
        else:
          alt_po = re.search(r"발주서[^\w]*번호[^\w]*([A-Za-z0-9]+)", full_text)
          if alt_po:
            po_no = alt_po.group(1).strip()

        disp_order = order_no if order_no else "미인식"
        disp_date = date if date else "미인식"
        disp_vendor = vendor if vendor else "업체명확인필요"
        disp_po = po_no if po_no else "미인식"

        new_filename = (
            f"{disp_order}_{disp_date}_{disp_vendor}_{disp_po}.{file_ext}"
        )

        zip_file.writestr(new_filename, file_bytes)

        processed_results.append({
            "original_name": file.name,
            "new_name": new_filename,
            "order_no": disp_order,
            "date": disp_date,
            "vendor": disp_vendor,
            "po_no": disp_po,
            "file_bytes": file_bytes,
            "ext": file_ext,
        })

      except Exception as e:
        st.error(f"'{file.name}' 처리 중 오류 발생: {e}")

      progress_bar.progress((idx + 1) / len(target_files))

  st.success("🎉 모든 파일 분석이 완료되었습니다!")

  zip_buffer.seek(0)
  st.download_button(
      label="📦 변환된 모든 파일 한 번에 다운로드 (ZIP)",
      data=zip_buffer,
      file_name="인수검사서_일괄변환.zip",
      mime="application/zip",
      type="primary",
  )

  st.markdown("---")
  st.markdown("### 📊 파일별 상세 결과 및 개별 다운로드")

  for res in processed_results:
    with st.expander(
        f"📁 기존 파일: {res['original_name']} ➔ 변경: {res['new_name']}",
        expanded=True,
    ):
      col1, col2 = st.columns([3, 1])
      with col1:
        st.write(
            f"**수주번호:** `{res['order_no']}` | **의뢰일자:**"
            f" `{res['date']}` | **업체명:** `{res['vendor']}` | **발주서번호:**"
            f" `{res['po_no']}`"
        )
      with col2:
        st.download_button(
            label="💾 변경된 파일 다운로드",
            data=res["file_bytes"],
            file_name=res["new_name"],
            mime=f"application/{res['ext']}",
            key=res["original_name"],
        )