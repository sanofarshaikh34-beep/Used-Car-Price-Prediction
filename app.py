import joblib
import streamlit as st
import pandas as pd
import textwrap

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CarWise | Used Car Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. COLOR PALETTE
# ==================================================

DARK_RED = "#35151C"
RED = "#7F2637"
RED_LIGHT = "#A13B4D"
GOLD = "#E5C18A"
GOLD_LIGHT = "#F4E4C8"
CREAM = "#FBF7F5"
WHITE = "#FFFFFF"
TEXT = "#3B2730"
MUTED = "#79656B"
BORDER = "#E9DADC"

# ==================================================
# 3. LOAD DATASET
# ==================================================

try:
    df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
    df.columns = df.columns.str.strip().str.lower()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

# ==================================================
# 4. LOAD TRAINED MODEL
# ==================================================

try:
    saved_model = joblib.load("car_price_model.pkl")
    car_model = saved_model["model"]
    model_mae = saved_model["mae"]
    model_r2 = saved_model["r2"]
    model_name = saved_model.get("model_name", "Machine Learning Regressor")
    model_features = saved_model.get("features", [])
    CURRENT_YEAR = saved_model.get("current_year", 2026)
except Exception as e:
    car_model = None
    model_mae = None
    model_r2 = None
    model_name = "Not Available"
    model_features = []
    CURRENT_YEAR = 2026
    model_error = str(e)

# ==================================================
# 5. CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #FBF7F5 0%, #F6EEEE 100%);
    color: #3B2730;
    font-family: 'Segoe UI', sans-serif;
}
.block-container {
    max-width: 1450px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #35151C !important;
}
.stApp [data-testid="stCaptionContainer"] p {
    color: #79656B !important;
}
[data-testid="stHeader"] {
    background: linear-gradient(90deg, #35151C, #7F2637, #A13B4D);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #35151C 0%, #4B202A 55%, #642331 100%);
    border-right: 1px solid #80505A;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] span {
    color: #F5E9E7 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 8px;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 11px;
    padding: 10px 12px;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(229,193,138,0.14);
    border-color: #E5C18A;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(100deg, #E5C18A, #F4E4C8);
    border: 1px solid #F4E4C8;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
    color: #35151C !important;
    font-weight: 700;
}
.hero-banner {
    background: linear-gradient(115deg, #35151C 0%, #602331 65%, #7F2637 100%);
    padding: 32px;
    border: 1px solid #80505A;
    border-radius: 20px;
    margin-bottom: 24px;
    box-shadow: 0 10px 28px rgba(53,21,28,0.13);
}
.hero-banner h1 {
    color: #FFFFFF !important;
    font-size: 32px;
    margin: 0 0 10px 0;
}
.hero-banner p {
    color: #F4E4C8 !important;
    font-size: 16px;
    margin: 0;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E9DADC;
    border-top: 3px solid #E5C18A;
    border-radius: 14px;
    padding: 19px;
    box-shadow: 0 4px 16px rgba(53,21,28,0.045);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #79656B !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #35151C !important;
    font-weight: 800;
}
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(105deg, #7F2637, #A13B4D);
    color: #FFFFFF !important;
    border: 1px solid #7F2637;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 700;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #E5C18A;
    color: #35151C !important;
    border-color: #E5C18A;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: #FFFFFF !important;
    color: #35151C !important;
    border: 1px solid #DCC8CC;
    border-radius: 9px;
}
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #FBF7F5 !important;
    color: #7F2637 !important;
    border: 1px solid #E9DADC !important;
}
[data-baseweb="popover"], [data-baseweb="popover"] > div, [role="listbox"] {
    background-color: #FBF7F5 !important;
}
[role="option"], [role="option"] span {
    background-color: #FBF7F5 !important;
    color: #7F2637 !important;
}
[data-testid="stDataFrame"] {
    background: #FFFFFF;
    border: 1px solid #E9DADC;
    border-radius: 11px;
    overflow: hidden;
}
[data-testid="stAlert"] {
    border-radius: 11px;
}
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E9DADC;
    border-radius: 10px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #35151C !important;
}
hr {
    border-color: #E9DADC;
}
[data-testid="stSidebar"] hr {
    border-color: #80505A;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 6. HELPER FUNCTIONS
# ==================================================

def render_html(html):
    """Render HTML after removing indentation from multiline strings."""
    st.markdown(
        textwrap.dedent(html).strip(),
        unsafe_allow_html=True
    )


def go_to_dataset_explorer():
    st.session_state["selected_page"] = "Dataset Explorer"


def page_header(title, subtitle):
    render_html(f"""
    <div class="hero-banner">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """)


def info_card(icon, title, description):
    render_html(f"""
    <div style="
        background:#FFFFFF;
        border:1px solid #E9DADC;
        border-top:3px solid #E5C18A;
        border-radius:15px;
        padding:22px;
        min-height:145px;
        box-shadow:0 4px 16px rgba(53,21,28,0.045);
    ">
        <div style="font-size:28px;margin-bottom:10px;">{icon}</div>
        <div style="
            color:#35151C !important;
            font-size:18px;
            font-weight:750;
            margin-bottom:7px;
        ">{title}</div>
        <div style="
            color:#79656B !important;
            font-size:14px;
            line-height:1.6;
        ">{description}</div>
    </div>
    """)


# ==================================================
# 7. DATASET INFORMATION
# ==================================================

has_price = "selling_price" in df.columns
has_name = "name" in df.columns
has_year = "year" in df.columns
has_fuel = "fuel" in df.columns
has_km = "km_driven" in df.columns

# ==================================================
# 8. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    render_html("""
    <div style="padding:12px 0 20px 0;">
        <div style="font-size:36px;margin-bottom:8px;">🚘</div>
        <h2 style="
            color:#FFFFFF !important;
            font-size:29px;
            margin:0;
            letter-spacing:-0.5px;
        ">CarWise</h2>
        <p style="
            color:#E5C18A !important;
            font-size:10px;
            letter-spacing:1.6px;
            margin-top:6px;
        ">USED CAR INTELLIGENCE</p>
    </div>
    """)

    render_html("""
    <p style="
        font-size:11px;
        letter-spacing:1.5px;
        font-weight:700;
        color:#F5E9E7 !important;
    ">MAIN MENU</p>
    """)

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Dataset Explorer",
            "Data Analysis",
            "Price Prediction",
            "Model Performance",
            "About Project"
        ],
        key="selected_page",
        label_visibility="collapsed"
    )

    st.markdown("---")

    model_status = (
        "● ML Model Ready"
        if car_model is not None
        else "● Model Not Available"
    )

    render_html(f"""
    <div style="
        background:rgba(255,255,255,0.055);
        border:1px solid rgba(229,193,138,0.25);
        border-radius:12px;
        padding:14px;
    ">
        <p style="
            color:#E5C18A !important;
            font-size:11px;
            font-weight:700;
            margin:0;
        ">PROJECT STATUS</p>
        <p style="
            color:#F5E9E7 !important;
            font-size:12px;
            margin:7px 0 0 0;
        ">{model_status}</p>
        <p style="
            color:#D9C5C9 !important;
            font-size:11px;
            margin:5px 0 0 0;
        ">CarWise Academic Project</p>
    </div>
    """)

    render_html("""
    <p style="
        color:#C9AEB4 !important;
        font-size:11px;
        margin-top:20px;
    ">CarWise | Academic Project</p>
    """)

# ==================================================
# 9. HOME PAGE
# ==================================================

if page == "Home":
    page_header(
        "Used Car Price Prediction",
        "Explore vehicle data and understand the used-car market."
    )

    st.markdown("### Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Total Vehicles", f"{len(df):,}")

    with c2:
        if has_price:
            st.metric("Average Price", f"₹{df['selling_price'].mean():,.0f}")
        else:
            st.metric("Average Price", "N/A")

    with c3:
        if has_name:
            st.metric("Car Models", f"{df['name'].nunique():,}")
        else:
            st.metric("Car Models", "N/A")

    with c4:
        if has_year:
            st.metric("Model Years", f"{df['year'].nunique():,}")
        else:
            st.metric("Model Years", "N/A")

    st.write("")
    left, right = st.columns([1.45, 1])

    with left:
        st.markdown("### Welcome to CarWise")
        st.write(
            "A data-driven application for exploring used cars, "
            "analysing prices and estimating vehicle prices using "
            "machine learning."
        )
        st.markdown("""
        - **Explore:** Search vehicle records.
        - **Analyse:** Understand price and fuel trends.
        - **Predict:** Estimate a used car's price.
        - **Evaluate:** Review model performance metrics.
        """)
        st.button(
            "Explore Dataset →",
            use_container_width=True,
            key="home_explore_dataset_button",
            on_click=go_to_dataset_explorer
        )

    with right:
        info_card(
            "🤖",
            "Machine Learning",
            f"CarWise uses a trained {model_name} model "
            "to estimate used-car prices."
        )

    if has_price:
        st.markdown("### Price Snapshot")
        a, b = st.columns(2)

        with a:
            st.markdown("**Lower-priced vehicles**")
            st.dataframe(
                df.nsmallest(5, "selling_price"),
                use_container_width=True,
                hide_index=True
            )

        with b:
            st.markdown("**Higher-priced vehicles**")
            st.dataframe(
                df.nlargest(5, "selling_price"),
                use_container_width=True,
                hide_index=True
            )

# ==================================================
# 10. DATASET EXPLORER
# ==================================================

elif page == "Dataset Explorer":
    page_header(
        "Dataset Explorer",
        "Browse, search and understand your vehicle records."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Total Records", f"{df.shape[0]:,}")

    with c2:
        st.metric("Total Columns", df.shape[1])

    with c3:
        st.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")

    st.markdown("### Search Vehicle Records")
    search_text = st.text_input(
        "Search dataset",
        placeholder="Search by car name, fuel, year or other details..."
    )

    filtered_df = df.copy()

    if search_text:
        matches = filtered_df.astype(str).apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                na=False
            )
        ).any(axis=1)
        filtered_df = filtered_df[matches]

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records")
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("View Column Information"):
        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Missing Values": df.isnull().sum().values
        })
        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )

# ==================================================
# 11. DATA ANALYSIS
# ==================================================

elif page == "Data Analysis":
    page_header(
        "Data Analysis",
        "Discover patterns and trends in used-car prices."
    )

    if has_price:
        left, right = st.columns(2)

        with left:
            st.markdown("### Selling Price Distribution")
            price_data = df["selling_price"].dropna()

            if not price_data.empty:
                histogram = pd.cut(
                    price_data,
                    bins=10,
                    duplicates="drop"
                ).value_counts().sort_index()
                histogram.index = histogram.index.astype(str)
                st.bar_chart(histogram)

        with right:
            st.markdown("### Average Price by Fuel")

            if has_fuel:
                fuel_prices = (
                    df.groupby("fuel")["selling_price"]
                    .mean()
                    .sort_values(ascending=False)
                )
                st.bar_chart(fuel_prices)
            else:
                st.info("Fuel column is not available.")

        st.markdown("### Average Price by Manufacturing Year")

        if has_year:
            yearly_prices = (
                df.groupby("year")["selling_price"]
                .mean()
                .sort_index()
            )
            st.line_chart(yearly_prices)

        st.markdown("### Average Price by Transmission")

        if "transmission" in df.columns:
            transmission_prices = (
                df.groupby("transmission")["selling_price"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(transmission_prices)
    else:
        st.warning("The selling_price column was not found.")

    st.markdown("### Dataset Summary")
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

# ==================================================
# 12. PRICE PREDICTION
# ==================================================

elif page == "Price Prediction":
    page_header(
        "Price Prediction",
        "Enter vehicle details to estimate the used car price."
    )

    info_card(
        "🚘",
        "Vehicle Details",
        "Select your car model and enter vehicle details "
        "to get an estimated selling price."
    )

    st.write("")

    if car_model is None:
        st.error(
            "Trained model not found. Check that car_price_model.pkl "
            "is present in the project folder."
        )

    elif has_name:
        with st.form("prediction_form"):
            left, right = st.columns(2)

            with left:
                car_options = sorted(
                    df["name"].dropna().astype(str).unique().tolist()
                )
                car_name = st.selectbox("Select Car Model", car_options)

                year = st.number_input(
                    "Manufacturing Year",
                    min_value=1990,
                    max_value=int(CURRENT_YEAR),
                    value=min(2020, int(CURRENT_YEAR)),
                    step=1
                )

                km_driven = st.number_input(
                    "Kilometres Driven",
                    min_value=0,
                    value=30000,
                    step=1000
                )

                fuel_options = (
                    sorted(df["fuel"].dropna().astype(str).unique().tolist())
                    if "fuel" in df.columns else ["Petrol"]
                )
                fuel = st.selectbox("Fuel Type", fuel_options)

            with right:
                transmission_options = (
                    sorted(df["transmission"].dropna().astype(str).unique().tolist())
                    if "transmission" in df.columns else ["Manual"]
                )
                transmission = st.selectbox("Transmission", transmission_options)

                seller_options = (
                    sorted(df["seller_type"].dropna().astype(str).unique().tolist())
                    if "seller_type" in df.columns else ["Individual"]
                )
                seller_type = st.selectbox("Seller Type", seller_options)

                owner_options = (
                    sorted(df["owner"].dropna().astype(str).unique().tolist())
                    if "owner" in df.columns else ["First Owner"]
                )
                owner = st.selectbox("Previous Owners", owner_options)

            submitted = st.form_submit_button(
                "Estimate Car Price",
                use_container_width=True
            )

        if submitted:
            try:
                # Feature engineering must match the training script.
                brand = str(car_name).strip().split()[0]
                car_age = max(1, int(CURRENT_YEAR) - int(year))
                km_per_year = float(km_driven) / car_age

                input_data = pd.DataFrame([{
                    "name": car_name,
                    "brand": brand,
                    "year": int(year),
                    "car_age": car_age,
                    "km_driven": int(km_driven),
                    "km_per_year": km_per_year,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner
                }])

                if not model_features:
                    raise ValueError(
                        "Model feature list is empty in car_price_model.pkl."
                    )

                input_data = input_data[model_features]
                predicted_price = float(car_model.predict(input_data)[0])
                predicted_price = max(0, predicted_price)

                st.success("Price estimate generated successfully!")
                st.markdown("### Estimated Car Price")

                render_html(f"""
                <div style="
                    background:#FBF7F5;
                    border:2px solid #7F2637;
                    padding:28px;
                    border-radius:16px;
                    text-align:center;
                    margin:15px 0;
                ">
                    <div style="
                        font-size:16px;
                        color:#79656B !important;
                    ">Estimated Selling Price</div>
                    <div style="
                        font-size:34px;
                        font-weight:700;
                        color:#7F2637 !important;
                    ">₹ {predicted_price:,.0f}</div>
                    <div style="
                        font-size:13px;
                        color:#3B2730 !important;
                    ">CarWise Machine Learning Estimate</div>
                </div>
                """)

                st.caption(
                    "This is an approximate estimate based on "
                    "historical dataset patterns. Actual market "
                    "prices may vary."
                )

                with st.expander("View Vehicle Details"):
                    submitted_details = pd.DataFrame([{
                        "Car Name": car_name,
                        "Brand": brand,
                        "Year": year,
                        "Car Age": car_age,
                        "Kilometres Driven": km_driven,
                        "KM Per Year": round(km_per_year, 2),
                        "Fuel": fuel,
                        "Transmission": transmission,
                        "Seller Type": seller_type,
                        "Previous Owners": owner
                    }])
                    st.dataframe(
                        submitted_details,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.error("Required dataset columns are missing.")

# ==================================================
# 13. MODEL PERFORMANCE
# ==================================================

elif page == "Model Performance":
    page_header(
        "Model Performance",
        "Understand the algorithm and its actual evaluation results."
    )

    st.markdown("### Trained Machine Learning Model")

    if car_model is not None:
        m1, m2 = st.columns(2)

        with m1:
            st.metric(
                "Mean Absolute Error (MAE)",
                f"₹{model_mae:,.2f}" if model_mae is not None else "N/A"
            )

        with m2:
            st.metric(
                "R² Score",
                f"{model_r2:.4f}" if model_r2 is not None else "N/A"
            )

        st.success(f"{model_name} model is trained and loaded.")
        st.caption(
            "MAE measures the average absolute prediction error "
            "in rupees. Lower is better. R² measures how well "
            "the model explains price variation."
        )

        if model_r2 is not None:
            st.markdown("### R² Score Percentage")
            st.progress(min(max(float(model_r2), 0.0), 1.0))
            st.write(f"Test R² Score: {model_r2 * 100:.2f}%")
    else:
        st.warning(
            "Model metrics are not available. Check that "
            "car_price_model.pkl is present and valid."
        )

    st.markdown("### Algorithm Used")
    info_card(
        "🌲",
        model_name,
        "An ensemble-based machine learning algorithm "
        "that combines multiple decision trees to estimate "
        "used-car selling prices."
    )

    st.markdown("### Evaluation Metrics")
    c1, c2 = st.columns(2)

    with c1:
        info_card(
            "📉",
            "Mean Absolute Error (MAE)",
            "Measures the average absolute difference between "
            "actual and predicted prices. Lower is better."
        )

    with c2:
        info_card(
            "🎯",
            "R² Score",
            "Measures how well the model explains price variation. "
            "A score closer to 1 is generally better."
        )

# ==================================================
# 14. ABOUT PROJECT
# ==================================================

elif page == "About Project":
    page_header(
        "About CarWise",
        "Used Car Price Prediction System | Academic Project"
    )

    st.markdown("### Project Objective")

    render_html("""
    <div style="
        background:#FFFFFF;
        border:1px solid #E9DADC;
        border-left:5px solid #7F2637;
        border-radius:14px;
        padding:23px;
        box-shadow:0 4px 16px rgba(53,21,28,0.04);
    ">
        <p style="color:#3B2730 !important;margin:0;">
            The objective of this project is to explore used-car data
            and estimate vehicle selling prices based on features such
            as car model, brand, manufacturing year, kilometres driven,
            fuel type, transmission, seller type and previous ownership.
        </p>
    </div>
    """)

    st.markdown("### Technologies Used")

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "Scikit-learn",
            "Joblib",
            "Streamlit",
            "Data Visualization"
        ],
        "Purpose": [
            "Programming language",
            "Dataset handling and analysis",
            "Machine learning model development",
            "Saving and loading the trained model",
            "Interactive web application",
            "Charts and visual insights"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Dataset Information")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Total Columns", len(df.columns))

    st.markdown("### Machine Learning Algorithms")
    st.markdown("""
    - Random Forest Regressor
    - Extra Trees Regressor
    - RandomizedSearchCV for hyperparameter tuning
    """)

    st.markdown("### Future Scope")
    st.markdown("""
    - Compare additional regression algorithms.
    - Improve predictions using additional vehicle features.
    - Update the dataset with recent market data.
    - Improve model performance through hyperparameter tuning.
    - Deploy the application for online use.
    """)

    st.caption(
        "CarWise | Used Car Price Prediction System | Academic Project"
    )
