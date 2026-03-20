"""
streamlit_app.py
"""
import streamlit as st 
import requests
import pandas as pd 

api_url = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="Product Intelligence Engine",
    page_icon="🔍",
    layout ="wide"
)

st.title("🔍 PRODUCT INTELLIGENCE ENGINE")

tab1, tab2, tab3, tab4 = st.tabs(["Similar Product Search","Category Predictor","Cluster Explorer","Analytics"])

# Tab 1 : Search

with tab1:
    st.subheader("Search Similar Products")
    query=st.text_input("Enter product name", placeholder="e.g. --> apple iphone 8 plus 64gb")
    top_k= st.slider("Number of results", min_value=1,max_value=50,value=10)

    if st.button("Search", key="search_btn"):
        if query.strip():
            with st.spinner("Searching..."):
                try:
                    res=requests.get(f"{api_url}/search",params={"query":query,"top_k":top_k},timeout=10)
                    if res.status_code==200:
                        data = res.json()
                        results=data.get("results",[])
                        st.success(f"Found {data['total']} results for '{data['query']}'")
                        df = pd.DataFrame(results)

                        # Ensure columns exist
                        if not df.empty:
                            df["similarity_score"] = df["similarity_score"].astype(float).round(4)

                            df = df.rename(columns={
                                "product_title": "Product Title",
                                "category_label": "Category",
                                "similarity_score": "Similarity Score"
                            })

                            df = df[["Product Title", "Category", "Similarity Score"]]
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("No results found.")
                    else:
                        st.error(f"API error: {res.status_code}")
                
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the backend is running on port 8000")
        else:
            st.warning("Please enter a search query.")

# Tab 2 : Predict
with tab2:
    st.subheader("Predict Product Category")
    title = st.text_input("Enter product title", placeholder="e.g. samsung 55 inch 4k smart tv")
    if st.button("Predict", key="predict_btn"):
        if title.strip():
            with st.spinner("Predicting..."):
                try:
                    res = requests.post(
                        f"{api_url}/predict",
                        json={"title": title},
                        timeout=10
                    )
                    if res.status_code == 200:
                        data = res.json()
                        predictions = data.get("predictions", [])
                        if predictions:
                            df = pd.DataFrame(predictions)
                            # Ensuring this to be numeric + clean
                            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
                            # Sorting by confidence
                            df = df.sort_values(by="confidence", ascending=False)
                            df["confidence"] = df["confidence"].round(4)
                            df = df.rename(columns={
                                "category": "Category",
                                "confidence": "Confidence"
                            })
                            st.success("Top Category Predictions:")
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("No category predictions found.")
                    else:
                        st.error(f"API error: {res.status_code}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the backend is working fine on port 8000")
        else:
            st.warning("Please enter a product title.")

# Tab 3 : Cluster Explorer

with tab3:
    st.subheader("Explorer Product Clusters")
    cluster_id = st.number_input("Enter Cluster ID", min_value=0, max_value=500,value=0,step=1)
    top_k_cluster=st.slider("Products to show", min_value=-5,max_value=100,value=20)
    if st.button("Explorer", key="cluster_btn"):
        with st.spinner("Fetching cluster..."):
            try:
                res=requests.get(f"{api_url}/cluster/{cluster_id}",params={"top_k":top_k_cluster},timeout=10)
                if res.status_code==200:
                    data=res.json()
                    st.success(f"Cluster{data['cluster_id']} - {data['total']} products")
                    df=pd.DataFrame(data["products"])
                    df.columns=["product_title", "category"]
                    st.dataframe(df,use_container_width=True)

                elif res.status_code==404:
                    st.warning("Cluster not found . Try a different Id.")
                else:
                    st.error(f"API error: {res.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the backend is working fine on port 8000")

# Tab 4: Analytics

with tab4:
    st.subheader("Dataset Analytics")
    with st.spinner("Loading Analytics..."):
        try:
            res=requests.get(f"{api_url}/analytics",timeout=10)
            if res.status_code==200:
                data =res.json()
                col1,col2=st.columns(2)
                col1.metric("Total Products",f"{data['total_products']:,}")
                col2.metric("Total Clusters",f"{data['total_clusters']:,}")
                st.markdown("### Category Distribution")
                cat_df=pd.DataFrame(data["category_distribution"].items(),columns=(["Category","Count"])).sort_values("Count",ascending=False)

                st.bar_chart(cat_df.set_index("Category"))
                st.dataframe(cat_df,use_container_width=True)
            else: 
                st.error(f"API error: {res.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the backend is working fine on port 8000")