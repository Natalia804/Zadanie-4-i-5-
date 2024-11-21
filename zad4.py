import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px

# Funkcja pomocnicza do oceny modelu
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return {
        "Accuracy (Train)": accuracy_score(y_train, y_train_pred),
        "Accuracy (Test)": accuracy_score(y_test, y_test_pred),
        "Precision (Test)": precision_score(y_test, y_test_pred),
        "Recall (Test)": recall_score(y_test, y_test_pred),
        "Confusion Matrix (Test)": confusion_matrix(y_test, y_test_pred),
    }

# Nagłówek aplikacji
st.title("Analiza modeli decyzyjnych w Pythonie")

# Automatyczne ładowanie danych z osadzonego pliku CSV
@st.cache_data
def load_data():
    file_path = "dane/zad3_Airline.csv"  
    data = pd.read_csv(file_path, sep=';')
    return data

# Wczytanie danych
data = load_data()


# Czyszczenie i przygotowanie danych
data['Customer.Type'].fillna('Unknown', inplace=True)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Gate.location'].fillna(data['Gate.location'].median(), inplace=True)
data['Arrival.Delay.in.Minutes'].fillna(data['Arrival.Delay.in.Minutes'].median(), inplace=True)

st.write("Podgląd danych:", data.head())

# Przetwarzanie zmiennych kategorycznych 
categorical_cols = ['Gender', 'Customer.Type', 'Type.of.Travel', 'Class']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
data['satisfaction'] = data['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)


pokaz_dane = st.checkbox("Pokaż dane po przetworzeniu zmiennych kategorycznych")

# Wyświetlanie danych w zależności od stanu toggle
if pokaz_dane:
    st.write("")
    st.dataframe(data)
else:
    st.write("")

# Podział na X i y
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


st.write("")
st.divider()
# Proste drzewo decyzyjne
st.subheader("Proste drzewo decyzyjne")
criterion = st.selectbox("Wybierz regułę klasyfikacyjną", ["gini", "entropy"])
max_depth = st.slider("Maksymalna głębokość drzewa", min_value=2, max_value=10, value=5)

dt_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
dt_model.fit(X_train, y_train)

# Wizualizacja drzewa decyzyjnego (przesunięta bezpośrednio pod suwakiem)
st.subheader("Wizualizacja drzewa decyzyjnego")
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(dt_model, feature_names=X.columns, class_names=["Neutral/Dissatisfied", "Satisfied"], filled=True, ax=ax)
st.pyplot(fig)

# Ocena modelu drzewa
dt_metrics = evaluate_model(dt_model, X_train, X_test, y_train, y_test)

# Wizualizacja wyników drzewa decyzyjnego w bardziej atrakcyjny sposób
st.write("### Wyniki drzewa decyzyjnego")

# Dodanie kolumn dla wskaźników
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Dokładność (Test)", value=f"{dt_metrics['Accuracy (Test)']:.2f}")

with col2:
    st.metric(label="Precyzja (Test)", value=f"{dt_metrics['Precision (Test)']:.2f}")

with col3:
    st.metric(label="Czułość (Test)", value=f"{dt_metrics['Recall (Test)']:.2f}")


# Wizualizacja macierzy konfuzji
conf_matrix = pd.DataFrame(
    dt_metrics["Confusion Matrix (Test)"], 
    index=["Neutral/Dissatisfied", "Satisfied"], 
    columns=["Predicted Neutral/Dissatisfied", "Predicted Satisfied"]
)
st.write("Macierz konfuzji (Test):")
st.dataframe(conf_matrix)


# Tworzenie tabeli porównawczej wyników dla różnych głębokości i reguł klasyfikacyjnych
st.write("")
st.subheader("Porównanie wyników drzewa decyzyjnego")

# Testowanie różnych parametrów drzewa
comparison_results = []
for criterion in ["gini", "entropy"]:
    for depth in range(2, 11):  # Głębokości od 2 do 10
        model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        comparison_results.append({
            "Criterion": criterion,
            "Max Depth": depth,
            "Accuracy (Test)": metrics["Accuracy (Test)"],
            "Precision (Test)": metrics["Precision (Test)"],
            "Recall (Test)": metrics["Recall (Test)"]
        })

# Konwersja wyników do DataFrame
comparison_df = pd.DataFrame(comparison_results)

# Tworzenie pivot table
pivoted_df = comparison_df.pivot_table(
    index="Max Depth",
    columns="Criterion",
    values=["Accuracy (Test)", "Precision (Test)", "Recall (Test)"]
)

# Wyświetlenie przekształconej tabeli
st.write("Tabela porównawcza wyników drzewa decyzyjnego (z podziałem na regułę klasyfikacyjną):")
st.dataframe(pivoted_df.style.format("{:.2f}"))

# Wizualizacja wyników w postaci wykresu
st.write("")
st.write("### Wykres porównawczy dokładności")
fig = px.line(
    comparison_df,
    x="Max Depth",
    y="Accuracy (Test)",
    color="Criterion",
    markers=True,
    title="Porównanie dokładności dla różnych głębokości drzewa i reguł klasyfikacyjnych"
)
st.plotly_chart(fig)



st.write("")
st.divider()
# Bagging
st.subheader("Bagging")
n_estimators = st.slider("Liczba drzew w baggingu", min_value=10, max_value=200, step=10, value=50)

bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
    n_estimators=n_estimators,
    random_state=42
)
bagging_model.fit(X_train, y_train)
bagging_metrics = evaluate_model(bagging_model, X_train, X_test, y_train, y_test)

# Wizualizacja wyników baggingu
st.write("### Wyniki baggingu")
st.metric("Dokładność (Test)", f"{bagging_metrics['Accuracy (Test)']:.2f}")
st.metric("Precyzja (Test)", f"{bagging_metrics['Precision (Test)']:.2f}")
st.metric("Czułość (Test)", f"{bagging_metrics['Recall (Test)']:.2f}")
st.write("")
st.divider()
# Porównanie wyników
st.subheader("Porównanie wyników")
comparison_df = pd.DataFrame({
    "Model": ["Drzewo Decyzyjne", "Bagging"],
    "Accuracy (Test)": [dt_metrics["Accuracy (Test)"], bagging_metrics["Accuracy (Test)"]],
    "Precision (Test)": [dt_metrics["Precision (Test)"], bagging_metrics["Precision (Test)"]],
    "Recall (Test)": [dt_metrics["Recall (Test)"], bagging_metrics["Recall (Test)"]]
})

# Wykres porównania wyników
fig = px.bar(
    comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Value"),
    x="Metric", y="Value", color="Model", barmode="group",
    title="Porównanie wyników modeli"
)
st.plotly_chart(fig)

##########


from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Lasy losowe
st.write("")
st.subheader("Lasy losowe")

# Suwaki dla parametrów lasów losowych
n_estimators_rf = st.slider("Liczba drzew w lesie losowym", min_value=10, max_value=200, step=10, value=100)
max_features_rf = st.selectbox("Liczba zmiennych w danym podziale (max_features)", ["sqrt", "log2", None])
max_depth_rf = st.slider("Maksymalna głębokość drzew w lesie losowym", min_value=2, max_value=20, step=1, value=None)

# Model lasów losowych
rf_model = RandomForestClassifier(
    n_estimators=n_estimators_rf,
    max_features=max_features_rf,
    max_depth=max_depth_rf,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Ewaluacja modelu lasów losowych
rf_metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test)

# Ważność zmiennych w lasach losowych
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Wyniki lasów losowych
st.write("### Wyniki lasów losowych")
st.metric("Dokładność (Test)", f"{rf_metrics['Accuracy (Test)']:.2f}")
st.metric("Precyzja (Test)", f"{rf_metrics['Precision (Test)']:.2f}")
st.metric("Czułość (Test)", f"{rf_metrics['Recall (Test)']:.2f}")

st.write("Ważność zmiennych w modelu lasów losowych:")
st.dataframe(feature_importances)

# Wizualizacja ważności zmiennych
fig = px.bar(feature_importances, x="Importance", y="Feature", orientation="h", title="Ważność zmiennych w lasach losowych")
st.plotly_chart(fig)


##########

from sklearn.ensemble import AdaBoostClassifier

# Boosting
st.write("")
st.subheader("Boosting")

# Suwaki dla hiperparametrów boostingu
n_estimators_boost = st.slider("Liczba estymatorów w boostingu", min_value=10, max_value=200, step=10, value=50)
learning_rate_boost = st.slider("Szybkość uczenia w boostingu (learning_rate)", min_value=0.01, max_value=1.0, step=0.01, value=0.1)

# Model boostingu
boost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=n_estimators_boost,
    learning_rate=learning_rate_boost,
    random_state=42
)
boost_model.fit(X_train, y_train)

# Ewaluacja modelu boostingu
boost_metrics = evaluate_model(boost_model, X_train, X_test, y_train, y_test)

# Wyniki boostingu
st.write("### Wyniki boostingu")
st.metric("Dokładność (Test)", f"{boost_metrics['Accuracy (Test)']:.2f}")
st.metric("Precyzja (Test)", f"{boost_metrics['Precision (Test)']:.2f}")
st.metric("Czułość (Test)", f"{boost_metrics['Recall (Test)']:.2f}")

# Ważność zmiennych w boostingu
boost_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": boost_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.write("Ważność zmiennych w modelu boostingu:")
st.dataframe(boost_importances)

# Wizualizacja ważności zmiennych
fig = px.bar(boost_importances, x="Importance", y="Feature", orientation="h", title="Ważność zmiennych w boostingu")
st.plotly_chart(fig)
