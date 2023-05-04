import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from snowflake.snowpark.functions import col, call_udf
from sklearn import metrics
import json

st.set_page_config(
    page_title="Snowflake Model Explorer",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/artemii-ustiukhin',
        'About': "From :flag-fi: with love!"
    }
)

### Create SnowFlake Session object

def create_session_object():
    conn = st.experimental_connection('snowpark')
    with conn.safe_session() as session:
        print("Test Snowflake session:")
        print(session.sql('select current_warehouse(), current_database(), current_schema()').collect())
        return session
    
### Helper functions

def describePowers(feeature_names, raw_powers):
    named_powers = []
    for powers_set in raw_powers:
        res = ""
        for i in range(len(feeature_names)):
            if (powers_set[i] == 0):
                continue
            if (res != ""):
                res += "+"
            res += feeature_names[i] + (("^" + str(powers_set[i])) if powers_set[i] > 1 else "")
        if res == "":
            res = "BIAS"
        named_powers.append(res)
    return named_powers

### Get Model Parameters

def get_model_data(session, model_name, model_name_full):
    # session.clear_imports()
    # session.clear_packages()
    # session.add_import('@dash_models/{0}'.format(model_name_full))
    # session.add_packages('pandas','joblib','scikit-learn==1.1.1', 'numpy')

    if model_name not in st.session_state.models_data:
        try:
            df = session.create_dataframe([model_name_full], schema=["Model Name"])
            data = df.select(call_udf("get_model", col("Model Name")).as_("UDF")).collect()
            if not data == None and len(data) > 0 and len(data[0]) > 0:
                model_data = json.loads(data[0][0])
                st.session_state.models_data[model_name] = model_data
                print("Updated st.session_state:")
                print(st.session_state)
                return True
        except Exception as e:
            print("Error on getting model data for {0}".format(model_name))
            print("Error:")
            print(str(e))
            return False

    return model_name in st.session_state.models_data

### Display Model Functions

def display_model_name(model_name, parent=st):
    if model_name == "SimpleLinearRegression":
        parent.title("Simple Linear Regression Model")
    elif model_name == "GradientBoostingRegression":
        parent.title("Gradient Boosting Regression Model")
    elif model_name == "ComplexLinearRegression":
        parent.title("Polynomial Linear Regression Model")
    elif model_name == "LinearSVC":
        parent.title("Linear Support Vector Classifier")
    elif model_name == "DecisionTree":
        parent.title("Decision Tree Model")

def display_model(model_data_received, model_name, parent=st):
    if model_data_received:
        model_data = st.session_state.models_data[model_name]

        if model_name == "SimpleLinearRegression":
            display_simple_linear_regression_model(model_data, parent)
        elif model_name == "GradientBoostingRegression":
            display_simple_linear_regression_model(model_data, parent)
        elif model_name == "ComplexLinearRegression":
            display_linear_regression_model(model_data, parent)
        elif model_name == "LinearSVC":
            display_svc_model(model_data, parent)
        elif model_name == "DecisionTree":
            display_desicion_tree_model(model_data, parent)

def display_simple_linear_regression_model(model_data, parent=st):
    # st.dataframe(model_data)
    model_data_frame = pd.DataFrame(data=model_data).sort_values("Coef", ascending=False)

    y_pos = list(range(0,4))
    model_features = model_data_frame["Feature_names"]
    data = model_data_frame["Coef"]

    fig, ax = plt.subplots()

    ax.barh(y_pos, data, align='center')
    ax.set_yticks(y_pos, labels=model_features)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature Name')
    # ax.set_title('Feature importance of SimpleLinearRegression model')

    parent.pyplot(fig)

def display_linear_regression_model(model_data_source, parent=st):
    powers = ["A", "B", "C", "D"]
    named_powers = describePowers(powers, model_data_source["Powers"])
    coeff_df = model_data_source["Coef"]
    model_data = { "Power_Names": named_powers, 
                    "Coefficients": coeff_df, 
                    "Feature_Names": model_data_source["Feature_names"] }
    # st.dataframe(model_data)
    data_for_frame = { "Power_Names": model_data["Power_Names"], 
                              "Coefficients": model_data["Coefficients"] }
    model_data_frame = pd.DataFrame(data=data_for_frame).sort_values("Coefficients", ascending=False)

    model_features = model_data_frame["Power_Names"]
    y_pos = list(range(0,len(model_features)))
    data = model_data_frame["Coefficients"]

    fig, ax = plt.subplots()

    ax.barh(y_pos, data, align='center')
    ax.set_yticks(y_pos, labels=model_features)
    ax.invert_yaxis()  # labels read top-to-bottom
    comment = ""
    for i in range(len(powers)):
        comment += "{0}: {1}; ".format(powers[i], model_data["Feature_Names"][i])
    
    ax.set_xlabel('Features Combination Importance')
    ax.set_ylabel('Features Combination')
    ax.set_title(comment)

    parent.pyplot(fig) 

def display_desicion_tree_model(model_data, parent=st):
    dot_data = model_data["Dot_data"]
    parent.graphviz_chart(dot_data)
    
def display_svc_model(model_data, parent=st):
    """
    Visualize the coefficients of a multi-class LinearSVC model.
    """
    feature_names = model_data["Feature_names"]
    class_names = model_data["Target_names"]

    coef = model_data["Coef"]
    n_classes = len(coef)
    n_features = len(coef[0])
    
    X_axis = np.arange(n_features)

    # create plot
    plt.figure(figsize=(15, 5))

    bar_width = (1 - 0.4) / n_classes
    shift = [-bar_width, 0, bar_width]

    for i in range(n_classes):
        plt.bar(X_axis + shift[i], coef[i], bar_width, label = class_names[i])
        # ax.bar_label(rects, label_type='center')

    plt.xticks(list(range(0,len(feature_names))), feature_names, rotation=60, ha='right')
    # plt.title('Feature Importance per class')
    plt.xlabel('Feature Name')
    plt.ylabel('Feature Importance')
    plt.legend(class_names)
    plt.grid(which="major", axis="y")

    parent.pyplot(plt)

### Get Model Predictions


def get_model_predictions(session, model_name, model_name_full, test_size):
    # session.clear_imports()
    # session.clear_packages()
    # session.add_import('@dash_models/{0}'.format(model_name_full))
    # session.add_packages('pandas','joblib','scikit-learn==1.1.1', 'numpy')

    if model_name not in st.session_state.models_predictions:
        if "Regression" in model_name:
            try:
                model_predictions = session.call('get_regression_model_prediction', model_name_full, test_size)
                json_predictions = json.loads(model_predictions)
                if not json_predictions == None:
                    st.session_state.models_predictions[model_name] = json_predictions
                    print("Updated st.session_state.models_predictions:")
                    print(st.session_state.models_predictions)
                    return True
            except Exception as e:
                print("Error on getting predictions for {0}".format(model_name))
                print("Error:")
                print(str(e))
                return False
        elif model_name == "LinearSVC" or model_name == "DecisionTree":
            try:
                df = session.create_dataframe([[model_name_full, test_size]], schema=["Model Name", "Test Size"])
                data = df.select(call_udf("get_class_model_prediction", col("Model Name"), col("Test Size")).as_("UDF")).collect()
                if not data == None and len(data) > 0 and len(data[0]) > 0:
                    json_predictions = json.loads(data[0][0])
                    st.session_state.models_predictions[model_name] = json_predictions
                    print("Updated st.session_state.models_predictions:")
                    print(st.session_state.models_predictions)
                    return True
            except Exception as e:
                print("Error on getting predictions for {0}".format(model_name))
                print("Error:")
                print(str(e))
                return False

    return model_name in st.session_state.models_predictions

# def get_class_model_prediction(session, model_name, test_size):
#     # session.clear_imports()
#     # session.clear_packages()
#     # session.add_import('@dash_models/{0}'.format(model_name))
#     # session.add_packages('pandas','joblib','scikit-learn==1.1.1', 'numpy')

#     df = session.create_dataframe([[model_name, test_size]], schema=["Model Name", "Test Size"])
#     data = df.select(call_udf("get_class_model_prediction", col("Model Name"), col("Test Size")).as_("UDF"))
#     predictions = json.loads(data.collect()[0][0])
#     return predictions

def display_training_report_class(model_name, parent):
    # model_prediction = get_class_model_prediction(session, model_name, 0.1)
    if not "models_predictions" in st.session_state or not model_name in st.session_state["models_predictions"]:
        return
    
    model_prediction = st.session_state.models_predictions[model_name]
    y_test = model_prediction["Test_target"]
    predicted = model_prediction["Predicted_target"]

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    parent.pyplot(plt)

    report = metrics.classification_report(y_test, predicted, output_dict=True)
    print(
        f"Classification report for {model_name}:\n"
        f"{report}\n"
    )
    class_report = { "0": report["0"], "1": report["1"], "2": report["2"] }
    accuracy_row = {'precision': '-', 'recall': '-', 'f1-score': report['accuracy'], 'support': report['macro avg']['support']}
    avg_report = { 'Accuracy': accuracy_row,'Macro avg': report['macro avg'], 'Weighted avg': report['weighted avg'] }
    class_report_frame = pd.DataFrame.from_dict(class_report, orient='index', columns=['precision', 'recall', 'f1-score', 'support'])
    avg_report_frame = pd.DataFrame.from_dict(avg_report, orient='index', columns=['precision', 'recall', 'f1-score', 'support'])
    # parent.dataframe(report)
    parent.subheader("Class recognition accuracy")
    parent.dataframe(class_report_frame)
    parent.subheader("Avg statistics")
    parent.dataframe(avg_report_frame)

def display_training_report_regression(model_name, parent, compare_values = None):
    # model_predictions = session.call('get_regression_model_prediction', model_name, 0.2)

    if not "models_predictions" in st.session_state or not model_name in st.session_state["models_predictions"]:
        return
    
    model_prediction = st.session_state.models_predictions[model_name]

    y_true = np.array(model_prediction["Test_target"])
    y_pred = np.array(model_prediction["Predicted_target"])

    col1, col2, col3 = parent.columns([1, 1, 1])

    # row 1
    r2 = metrics.r2_score(y_true, y_pred)
    col1.metric(label="R2 Score", value="{:.2f}".format(r2), delta=None if compare_values == None else "{:.4f}".format(r2 - compare_values["r2"]))

    mae = metrics.mean_absolute_error(y_true, y_pred)
    col2.metric(label="MAE", value="{:.2f}".format(mae), delta=None if compare_values == None else "{:.4f}".format(mae - compare_values["mae"]))

    medae = metrics.median_absolute_error(y_true, y_pred)
    col3.metric(label="MedAE", value="{:.2f}".format(medae), delta=None if compare_values == None else "{:.4f}".format(medae - compare_values["medae"]))

    # row 2 
    # d2_absolute_error_score
    d2 = metrics.d2_absolute_error_score(y_true, y_pred)
    col1.metric(label="D2 absolute error score", value="{:.2f}".format(medae), delta=None if compare_values == None else "{:.4f}".format(d2 - compare_values["d2"]))

    ev = metrics.explained_variance_score(y_true, y_pred)
    col2.metric(label="Explained variance", value="{:.2f}".format(ev), delta=None if compare_values == None else "{:.4f}".format(ev - compare_values["ev"]))

    mpl = metrics.mean_pinball_loss(y_true, y_pred)
    col3.metric(label="Mean pinball loss", value="{:.2f}".format(mpl), delta=None if compare_values == None else "{:.4f}".format(mpl - compare_values["mpl"]))

    parent.subheader("Cross-validated predictions")

    display = metrics.PredictionErrorDisplay.from_predictions(y_true=y_true, y_pred=y_pred)
    display.plot()
    parent.pyplot(plt) 

    return { "r2": r2, "mae": mae, "medae": medae, "d2": d2, "ev": ev, "mpl":mpl }

def retrain_model(session, model_name, params):
    try:
        if model_name == "SimpleLinearRegression":
            session.call('train_model_simple_linear_regression', model_name, params["test_size"])
        if model_name == "GradientBoostingRegression":
            session.call('train_model_gradient_boosting_regression', model_name, params["test_size"], 0.1, 100, 'squared_error')
        elif model_name == "ComplexLinearRegression":
            session.call('train_model_complex_linear_regression', model_name, params["test_size"], params["folds"])
        elif model_name == "LinearSVC":
            session.call('train_iris_prediction_model', model_name, params["test_size"])
        elif model_name == "DecisionTree":
            session.call('train_iris_prediction_model', model_name, params["test_size"])
    except Exception as e:
        print("Error on retraining model {0}".format(model_name))
        print("Error:")
        print(str(e))

def display_model_params_selector(model_name, parent=st):
    train_size = st.slider('Train split size', min_value=0.1, max_value=0.9, value=0.5, step=0.1)
    if model_name == "SimpleLinearRegression":
        return { "test_size": 1 - train_size }
    elif model_name == "GradientBoostingRegression":
        return { "test_size": 1 - train_size }
    elif model_name == "ComplexLinearRegression":
        number_of_folds = st.slider('Number of folds for cross-validation', 1, 30, 10)
        return { "test_size": 1 - train_size, "folds": number_of_folds }
    elif model_name == "LinearSVC":
        return { "test_size": 1 - train_size }
    elif model_name == "DecisionTree":
        return { "test_size": 1 - train_size }

def retrain_model_button(snowflake_session, model_name, train_params):
    st.session_state.model_retraining_count += 1

    if st.session_state.model_retraining_count > 5:
        st.error("Too many model retraining attempts per session. Limit is 5.")
        return

    retrain_model(snowflake_session, model_name, train_params)

    # Remove old model data from state
    if model_name in st.session_state.models_data:
        st.session_state.models_data.pop(model_name)
    if model_name in st.session_state.models_predictions:
        st.session_state.models_predictions.pop(model_name)

def get_models_list_from_snowflake():
    models_raw = session.sql('LIST @DASH_MODELS').collect()
    models_names = {model.name.replace("dash_models/", "").split(".", 1)[0] : model.name.replace("dash_models/", "") for model in models_raw}
    models_sizes = {model.name.replace("dash_models/", "").split(".", 1)[0] : model.size for model in models_raw}
    print("Models names:")
    print(models_names)
    return models_names, models_sizes

if 'model_retraining_count' not in st.session_state:
    st.session_state.model_retraining_count = 0

if 'models_data' not in st.session_state:
    st.session_state["models_data"] = { "Model_data_object": "Model_data_value" }

if 'models_predictions' not in st.session_state:
    st.session_state["models_predictions"] = { "Model_prediction_object": "Model_prediction_value" }

### Main body

# Create SnowFlake Session
session = create_session_object()


# Get Trained Models list 
models_names, models_sizes = get_models_list_from_snowflake()

while len(models_names) == 0:
    with st.spinner('Extracting models from SnowFlake...'):
            models_names, models_sizes = get_models_list_from_snowflake()

# Sidebar
st.sidebar.title("Visualize Model")

model_name = st.sidebar.selectbox("Choose a Model", models_names.keys())
model_name_full = models_names[model_name]

regression_models = dict(filter(lambda item: "Regression" in item[0], models_names.items()))
classification_models = dict(filter(lambda item: "Regression" not in item[0], models_names.items()))

st.sidebar.title("Regression Problem Models :chart_with_upwards_trend:")
st.sidebar.dataframe({ "Name" : regression_models.keys(), "Size" : [models_sizes[model] for model in regression_models.keys()] })

st.sidebar.title("Classification Problem Models :bar_chart:")
st.sidebar.dataframe({ "Name" : classification_models.keys(), "Size" : [models_sizes[model] for model in classification_models.keys()] })

# Model explorer tab

st.header("Welcome to SnowFlake Models PlayGround ≈@_@≈")

model_data_received = get_model_data(session, model_name, model_name_full)

model_explorer_tab, model_comparator_tab = st.tabs(["Model Explorer :open_book:", "Model Comparator :scales:"])

with model_explorer_tab:
    model_column, params_column = st.columns([4, 2])

    with model_column:
        model_column.subheader("Explore Model Entrails")
        display_model(model_data_received, model_name, model_column)

    with params_column:
        params_column.subheader("Play with Model Parameters")
        train_params = display_model_params_selector(model_name, params_column)

        params_column.button("Retrain", on_click=retrain_model_button, args=(session, model_name, train_params, ))
        # if retrain and not model_name == "":
        #     with st.spinner("Retraining the model..."):
        #         retrain_model(session, model_name, train_params)
                # display_model(model_data, model_name, model_column)

# Model comparator tab

with model_comparator_tab:
    model_comparator_tab.subheader("Compare Models Performance")
    column_model_1, _, column_model_2 = st.columns([4, 1, 4])

    with column_model_1:
        model_name_1 = column_model_1.selectbox("Choose a model to compare", models_names.keys())
        model_name_full_1 = models_names[model_name_1]
        model_1_received_predictions = get_model_predictions(session, model_name_1, model_name_full_1, 0.2)
        if model_name_1 and model_1_received_predictions:
            if model_name_1 == "LinearSVC" or model_name_1 == "DecisionTree":
                column_model_1.subheader("Confusion matrix for {0} model".format(model_name_1))
                display_training_report_class(model_name_1, column_model_1)
            elif "Regression" in model_name_1:
                column_model_1.subheader("{0} model metrics".format(model_name_1))
                model_results = display_training_report_regression(model_name_1, column_model_1)

    with column_model_2:
        model_names_2 = list(regression_models.keys()) if model_name_1 in regression_models else list(classification_models.keys())
        model_names_2.remove(model_name_1)
        model_name_2 = column_model_2.selectbox("Choose another model to compare", model_names_2)
        model_name_full_2 = models_names[model_name_2]
        model_2_received_predictions = get_model_predictions(session, model_name_2, model_name_full_2, 0.2)
        if model_name_2 and model_2_received_predictions:
            if model_name_2 == "LinearSVC" or model_name_2 == "DecisionTree":
                column_model_2.subheader("Confusion matrix for {0} model".format(model_name_2))
                display_training_report_class(model_name_2, column_model_2)
            elif "Regression" in model_name_2:
                column_model_2.subheader("{0} model metrics".format(model_name_1))
                display_training_report_regression(model_name_2, column_model_2, model_results)
