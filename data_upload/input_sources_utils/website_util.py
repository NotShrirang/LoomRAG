import bs4
import os
import requests
import streamlit as st
import sys
from vectordb import add_image_to_index, add_pdf_to_index
from data_upload.input_sources_utils import text_util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def data_from_website(clip_model, preprocess, text_embedding_model):
    st.title("Data from Website")
    website_url = st.text_input("Enter Website URL")
    if website_url:
        st.write(f"URL: {website_url}")
        if st.button("Extract and Add Data"):
            response = requests.get(website_url)
            if response.status_code == 200:
                st.success("Data Extracted Successfully")
            else:
                st.error("Invalid URL")

            soup = bs4.BeautifulSoup(response.content, features="lxml")
            images = soup.find_all("img")
            image_dict = []
            if not images:
                st.info("No Images Found!")
            else:
                st.info(f"Found {len(images)} Images")
                progress_bar = st.progress(0, f"Extracting Images... | 0/{len(images)}")
                cols = st.columns(5)
                for count, image in enumerate(images):
                    try:
                        image_url = image["src"].replace("//", "https://")
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            image_dict.append({"src": image_url, "content": response.content})
                            add_image_to_index(response.content, clip_model, preprocess)
                            len_image_dict = len(image_dict)
                            if len_image_dict <= 4:
                                with cols[len_image_dict - 1]:
                                    st.image(image_url, caption=image_url, use_container_width=True)
                            elif len_image_dict == 5:
                                with cols[4]:
                                    st.info(f"and more {len(images) - 4} images...")
                    except:
                        pass
                    progress_bar.progress((count + 1) / len(images), f"Extracting Images... | {count + 1}/{len(images)}")
                progress_bar.empty()

            main_content = soup.find('main')
            sample_text = main_content.text.strip().replace(r'\n', '')
            with st.spinner("Processing Text..."):
                text_util.process_text(main_content.text, text_embedding_model)
            st.success("Data Added to Database")
