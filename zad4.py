import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import plotly.express as px

# Funkcja pomocnicza do oceny modelu
def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    return {
        "Accuracy (Train)": accuracy_score(y_train, y_train_pred),
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_test_pred),
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

def uzupelnij_braki_kategoryczne(df, kolumny):
    """
    Uzupełnia brakujące wartości w kategorycznych kolumnach 
    losowymi wartościami z istniejących wartości w tych kolumnach.
    """
    for kolumna in kolumny:
        istniejące_wartosci = df[kolumna].dropna().unique()
        maska_brakow = df[kolumna].isna()
        liczba_brakow = maska_brakow.sum()
        losowe_wartosci = np.random.choice(istniejące_wartosci, size=liczba_brakow, replace=True)
        df.loc[maska_brakow, kolumna] = losowe_wartosci
    return df

# Czyszczenie i przygotowanie danych
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Arrival.Delay.in.Minutes'].fillna(data['Arrival.Delay.in.Minutes'].median(), inplace=True)
data = uzupelnij_braki_kategoryczne(data, ['Customer.Type', 'Gate.location'])

st.write("Podgląd danych:", data.head())

# Przetwarzanie zmiennych kategorycznych 
categorical_cols = ['Gender', 'Customer.Type', 'Type.of.Travel', 'Class']
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
data['satisfaction'] = data['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)

with st.expander("Dane po przetworzeniu zmiennych kategorycznych"):
    st.dataframe(data)

# Podział na X i y
X = data.drop(columns=['satisfaction'])
y = data['satisfaction']

# Podział na zbiory treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


st.divider()


# Proste drzewo decyzyjne
st.subheader("Proste drzewo decyzyjne")
criterion = st.selectbox("Wybierz regułę klasyfikacyjną", ["gini", "entropy"])
max_depth = st.slider("Maksymalna głębokość drzewa", min_value=2, max_value=20, value=5)

dt_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
dt_model.fit(X_train, y_train)

# Wizualizacja drzewa decyzyjnego
st.subheader(f"Wizualizacja drzewa decyzyjnego")
st.write(f"z zastosowaniem reguły klasyfikacyjnej `{criterion}` i maksymalną głębokością równą `{max_depth}`")
fig, ax = plt.subplots(figsize=(12, 8))

# Wykres drzewa
plot_tree(
    dt_model, 
    feature_names=X.columns, 
    class_names=["Neutral/Dissatisfied", "Satisfied"], 
    filled=True, 
    ax=ax)
st.pyplot(fig)

# Ocena modelu drzewa
dt_metrics = evaluate_model(dt_model, X_train, X_test, y_train, y_test)

# Wizualizacja wyników drzewa decyzyjnego w bardziej atrakcyjny sposób
st.write("#### Wyniki drzewa decyzyjnego")

# Dodanie kolumn dla wskaźników
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Dokładność", value=f"{dt_metrics['Accuracy']:.4f}")

with col2:
    st.metric(label="Precyzja", value=f"{dt_metrics['Precision']:.4f}")

with col3:
    st.metric(label="Czułość", value=f"{dt_metrics['Recall']:.4f}")


# Wizualizacja macierzy błędu
conf_matrix = pd.DataFrame(
    dt_metrics["Confusion Matrix"], 
    index=["Rzeczywiste Neutral/Dissatisfied", "Rzeczywiste Satisfied"], 
    columns=["Przewidziane Neutral/Dissatisfied", "Przewidziane Satisfied"]
)
st.write("Macierz błędów:")
st.dataframe(conf_matrix)


# Tworzenie tabeli porównawczej wyników dla różnych głębokości i reguł klasyfikacyjnych
st.write("")
st.subheader("Porównanie wyników drzewa decyzyjnego")

# Testowanie różnych parametrów drzewa
comparison_results = []
for criterion in ["gini", "entropy"]:
    for depth in range(2, 21):  # Głębokości od 2 do 10
        model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        comparison_results.append({
            "Criterion": criterion,
            "Max Depth": depth,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"]
        })

# Konwersja wyników do DataFrame
comparison_df = pd.DataFrame(comparison_results)

st.markdown(
    "Analiza została przeprowadzona na podstawie wyników uzyskanych dla różnych wartości **`max_depth`** oraz dwóch kryteriów podziału w drzewie decyzyjnym: **`entropy`** i **`gini`**. Dla każdej wartości głębokości drzewa oceniono trzy kluczowe metryki wydajności modelu:"
)

# Tworzenie pivot table
pivoted_df = comparison_df.pivot_table(
    index="Max Depth",
    columns="Criterion",
    values=["Accuracy", "Precision", "Recall"]
)

# Wyświetlenie przekształconej tabeli
st.write("Tabela porównawcza wyników drzewa decyzyjnego (z podziałem na regułę klasyfikacyjną):")
st.dataframe(pivoted_df.style.format("{:.4f}"))

# Wizualizacja wyników w postaci wykresu
st.write("")
st.write("### Wykres porównawczy dokładności")
fig = px.line(
    comparison_df,
    x="Max Depth",
    y="Accuracy",
    color="Criterion",
    markers=True,
    title="Porównanie dokładności dla różnych głębokości drzewa i reguł klasyfikacyjnych"
)
st.plotly_chart(fig)

st.markdown(
    """
    ### Wybór optymalnego drzewa decyzyjnego
    Na podstawie przeprowadzonej analizy, optymalny wybór drzewa decyzyjnego dla projektu analizy danych dotyczących zadowolenia klientów linii lotniczych jest następujący:
    - **`max_depth`:** 7
    - **Kryterium podziału:** `gini`

    **Powody wyboru:**

    1. **Najwyższa Dokładność (Accuracy):** Model osiąga najwyższą dokładność na zbiorze walidacyjnym i testowym przy **`max_depth = 7`** i kryterium **`gini`**.
    2. **Najwyższa Precyzja (Precision):** Minimalizuje fałszywe pozytywy, co jest istotne w zastosowaniach wymagających wysokiej wiarygodności przewidywań.
    3. **Wysoka Czułość (Recall):** Model skutecznie wykrywa większość pozytywnych przypadków, zapewniając dobrą zdolność generalizacji bez nadmiernego przeuczenia.
    """
)

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
st.write(f"dla liczby drzew w baggingu równą `{n_estimators}`")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Dokładność", value=f"{bagging_metrics['Accuracy']:.4f}")

with col2:
    st.metric(label="Precyzja", value=f"{bagging_metrics['Precision']:.4f}")

with col3:
    st.metric(label="Czułość", value=f"{bagging_metrics['Recall']:.4f}")


st.write("")

# Generowanie tabeli wyników baggingu
st.subheader("Tabela wyników Baggingu")

# Cache'owana funkcja do obliczania wyników dla różnych liczby drzew
@st.cache_data
def generate_bagging_table(X_train, X_test, y_train, y_test):
    results = []
    n_estimators_range = range(10, 201, 10)  # Liczba drzew od 10 do 200 co 10
    for n_estimators in n_estimators_range:
        # Tworzenie i trenowanie modelu Bagging
        bagging_model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
            n_estimators=n_estimators,
            random_state=42
        )
        bagging_model.fit(X_train, y_train)
        metrics = evaluate_model(bagging_model, X_train, X_test, y_train, y_test)
        
        # Zapis wyników dla każdej liczby drzew
        results.append({
            "Liczba Drzew": n_estimators,
            "Dokładność": metrics["Accuracy"],
            "Precyzja": metrics["Precision"],
            "Czułość": metrics["Recall"]
        })
    
    return pd.DataFrame(results)

# Generowanie wyników dla różnych liczby drzew
bagging_results_df = generate_bagging_table(X_train, X_test, y_train, y_test)

# Przekształcenie wyników w tabelę pivot
pivoted_bagging_table = bagging_results_df.pivot_table(
    index="Liczba Drzew",
    values=["Dokładność", "Precyzja", "Czułość"]
)

# Wyświetlenie tabeli w Streamlit
st.write("Tabela wyników Baggingu dla różnych liczby drzew:")
st.dataframe(pivoted_bagging_table.style.format("{:.4f}"))

st.markdown(
    """
    Zakres `10-60` Drzew:
    Znaczny wzrost dokładności i precyzji, co sugeruje, że zwiększenie liczby drzew w tym zakresie znacząco poprawia wydajność modelu.

    Zakres `70-200` Drzew:
    Metryki utrzymują się na stabilnym poziomie, z minimalnymi zmianami. Dalsze zwiększanie liczby drzew nie przynosi istotnych korzyści, a jedynie zwiększa złożoność modelu i czas trenowania.
    """
)

st.write(
    """
    Analiza wyników modelu Bagging wykazała, że liczba drzew 60 zapewnia optymalny balans między Dokładnością (0.895), Precyzją (0.9072) oraz Czułością (0.8800).
    Dalsze zwiększanie liczby drzew powyżej 60 nie przynosi znaczących korzyści, a jedynie minimalnie stabilizuje metryki na poziomie około 0.89-0.90.
    Ta konfiguracja modelu gwarantuje wysoką wydajność przy efektywnym wykorzystaniu zasobów obliczeniowych, jednocześnie minimalizując ryzyko przeuczenia.

"""
)

st.divider()

##########

# Porównanie wyników Proste Drzewo Decyzyjne vs Bagging
st.subheader("Porównanie wyników drzewa decyzyjnego vs bagging")

# Tworzenie DataFrame porównawczego
comparison_df = pd.DataFrame({
    "Model": ["Drzewo Decyzyjne", "Bagging"],
    "Accuracy": [dt_metrics["Accuracy"], bagging_metrics["Accuracy"]],
    "Precision": [dt_metrics["Precision"], bagging_metrics["Precision"]],
    "Recall": [dt_metrics["Recall"], bagging_metrics["Recall"]]
})

# Wyświetlenie tabeli porównawczej
st.write("Tabela porównawcza")
st.dataframe(comparison_df.style.format({
    "Accuracy": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}"
}))

# Wizualizacja porównania wyników jako wykres słupkowy
fig_compare = px.bar(
    comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Value"),
    x="Metric", y="Value", color="Model", barmode="group",
    title="Porównanie wyników modeli",
    labels={"Value": "Wartość Metryki", "Metric": "Metryka"}
)
st.plotly_chart(fig_compare)

st.markdown(
    """
    Na podstawie przeanalizowanych metryk drzewo decyzyjne wykazało się lepszą wydajnością i precyzją niż bagging, co oznacza, że drzewo decyzyjne prawidłowo klasyfikuje nieco więcej przypadków niż model Baggingu oraz wskazuje mniejszą liczbę fałszywych pozytywów.
    Model drzewa lepiej radzi sobie w sytuacjach wymagających wysokiej wiarygodności przewydiwań. W kontekście analizy zadowolenia klientów linii lotniczych, ważne jest, aby uniknąć fałszywych pozytywów (np. błędna klasyfikacja niezadowolonych klientów jako zadowolonych). Bagging byłby dobrym wyborem jeśli zależałoby na maksymalizacji wykrywania pozytywnych przypadków, czyli identyfikacji zadowolonych klientów. Natomiast w obecnym przypadku drzewo decyzyjne może być bardziej odpowiednim modelem. 
"""
)


##########
st.divider()
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
st.write(f"dla liczby drzew `{n_estimators_rf}`,  regułu podziału `{max_features_rf}` i głębokości `{max_depth_rf}`")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Dokładność", value=f"{rf_metrics['Accuracy']:.4f}")

with col2:
    st.metric(label="Precyzja", value=f"{rf_metrics['Precision']:.4f}")

with col3:
    st.metric(label="Czułość", value=f"{rf_metrics['Recall']:.4f}")


# Wizualizacja ważności zmiennych
fig = px.bar(feature_importances, x="Importance", y="Feature", orientation="h", title="Ważność zmiennych w lasach losowych")
st.plotly_chart(fig)



# Generowanie wyników dla różnych kombinacji parametrów lasów losowych
@st.cache_data
def generate_rf_table(X_train, X_test, y_train, y_test):
    results = []
    n_estimators_range = [10, 50, 100, 150, 200]
    max_features_range = ['sqrt', 'log2', None]
    max_depth_range = [5, 10, 15, 20]

    for n_estimators in n_estimators_range:
        for max_features in max_features_range:
            for max_depth in max_depth_range:
                rf_model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    max_depth=max_depth,
                    random_state=42
                )
                rf_model.fit(X_train, y_train)
                metrics = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
                results.append({
                    "Liczba Drzew (n_estimators)": n_estimators,
                    "Max Features": max_features,
                    "Max Depth": max_depth,
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"]
                })
    return pd.DataFrame(results)

# Generowanie tabeli wyników
rf_results_df = generate_rf_table(X_train, X_test, y_train, y_test)

# Wyświetlenie tabeli wyników
st.write("### Tabela wyników lasów losowych dla różnych kombinacji parametrów:")
st.dataframe(rf_results_df.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}))

# Wizualizacja wyników
st.write("### Wizualizacja wyników lasów losowych")
fig = px.parallel_coordinates(
    rf_results_df,
    dimensions=["Liczba Drzew (n_estimators)", "Max Features", "Max Depth"],
    color="Accuracy",
    title="Wpływ hiperparametrów na wyniki lasów losowych"
)
st.plotly_chart(fig)

st.markdown(
    """
    Główne wnioski:
    - Wyższa liczba drzew (**`n_estimators`**=200) daje najlepsze rezultaty, bez względu na pozostałe ustawienia.
    - Wartość **`max_depth`**=10 stabilizuje wyniki, podczas gdy 5 lub głębokość 20 nie zawsze poprawiają wyniki.
    - **`max_features`**= "sqrt" wydaje się być najlepszym wyborem w większości kombinacji.

"""
)

# Wybór najlepszych parametrów
best_rf = rf_results_df.loc[rf_results_df["Accuracy"].idxmax()]
st.write("### Najlepszy zestaw parametrów lasu losowego:")
st.json({
    "Liczba Drzew (n_estimators)": int(best_rf["Liczba Drzew (n_estimators)"]),
    "Max Features": best_rf["Max Features"],
    "Max Depth": best_rf["Max Depth"],
    "Accuracy": best_rf["Accuracy"],
    "Precision": best_rf["Precision"],
    "Recall": best_rf["Recall"]
})

st.markdown(
    """
    Dlaczego te parametry?

    - Liczba drzew: Większa liczba drzew zapewnia bardziej stabilne i dokładne przewidywania. Zbyt mała liczba drzew może prowadzić do niestabilnych wyników (np.**`n_estimators`**).
    - Liczba cech w podziale (sqrt): Ograniczenie cech losowanych w każdym węźle wprowadza różnorodność w drzewach, co zmniejsza ryzyko przeuczenia i poprawia zdolność generalizacji.
    - Maksymalna głębokość (**`max_depth`**): Ograniczenie głębokości pozwala uniknąć przeuczenia, co jest szczególnie istotne przy dużej liczbie drzew.

    """
)


# Ponowna analiza na podstawie najlepszych parametrów
optimal_rf_model = RandomForestClassifier(
    n_estimators=int(int(best_rf["Liczba Drzew (n_estimators)"])),
    max_features=best_rf["Max Features"],
    max_depth=int(best_rf["Max Depth"]),
    random_state=42
)
optimal_rf_model.fit(X_train, y_train)


st.divider()
##########
# Boostings
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
st.write(f"dla liczby estymatorów `{n_estimators_boost}` i szybkości uczenia`{learning_rate_boost}`")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Dokładność", value=f"{boost_metrics['Accuracy']:.4f}")

with col2:
    st.metric(label="Precyzja", value=f"{boost_metrics['Precision']:.4f}")

with col3:
    st.metric(label="Czułość", value=f"{boost_metrics['Recall']:.4f}")



# Ważność zmiennych w boostingu
boost_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": boost_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# Wizualizacja ważności zmiennych
fig = px.bar(boost_importances, x="Importance", y="Feature", orientation="h", title="Ważność zmiennych w boostingu")
st.plotly_chart(fig)



@st.cache_data
def generate_boosting_table(X_train, X_test, y_train, y_test):
    results = []
    n_estimators_range = [10, 50, 100, 150, 200]
    learning_rate_range = [0.01, 0.1, 0.2, 0.5, 1.0]

    for n_estimators in n_estimators_range:
        for learning_rate in learning_rate_range:
            boost_model = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=5),
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
            boost_model.fit(X_train, y_train)
            metrics = evaluate_model(boost_model, X_train, X_test, y_train, y_test)
            results.append({
                "Liczba Estymatorów (n_estimators)": n_estimators,
                "Learning Rate": learning_rate,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"]
            })
    return pd.DataFrame(results)

# Generowanie tabeli wyników boostingu
boosting_results_df = generate_boosting_table(X_train, X_test, y_train, y_test)

# Wyświetlenie tabeli wyników
st.write("### Tabela wyników boostingu dla różnych kombinacji parametrów:")
st.dataframe(boosting_results_df.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}))

# Wizualizacja wyników
fig = px.parallel_coordinates(
    boosting_results_df,
    dimensions=["Liczba Estymatorów (n_estimators)", "Learning Rate"],
    color="Accuracy",
    title="Wpływ hiperparametrów na wyniki boostingu"
)
st.plotly_chart(fig)

st.markdown(
    """
    Główne wnioski:
    
    - Najlepsze wyniki (najwyższe Accuracy) osiągnięto dla Learning rate = 0.01 przy wyższych liczbach estymatorów.
    
    - Wartości Learning rate większe niż 0.2 przy niskich liczbach estymatorów skutkują spadkiem dokładności, co może wskazywać na brak wystarczającej ilości iteracji do nauki.
    
    - Zbyt duże Learning rate (np. 1.0) prowadzą do spadku jakości modelu nawet przy większej liczbie estymatorów.

    """
)



# Wybór najlepszych parametrów
best_boost = boosting_results_df.loc[boosting_results_df["Accuracy"].idxmax()]
st.write("### Najlepszy zestaw parametrów boostingu:")
st.json({
    "Liczba Estymatorów (n_estimators)": best_boost["Liczba Estymatorów (n_estimators)"],
    "Learning Rate": best_boost["Learning Rate"],
    "Accuracy": best_boost["Accuracy"],
    "Precision": best_boost["Precision"],
    "Recall": best_boost["Recall"]
})

st.markdown(
    """
    Duża liczba estymatorów umożliwia modelowi precyzyjne dopasowanie do danych. Niskie tempo uczenia (**`Learning rate`**) zapewnia stabilność i unikanie nadmiernego dopasowania.

"""
)


# Tworzenie i trenowanie modelu z najlepszymi parametrami
optimal_boost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=int(best_boost["Liczba Estymatorów (n_estimators)"]),
    learning_rate=best_boost["Learning Rate"],
    random_state=42
)
optimal_boost_model.fit(X_train, y_train)

# Ważność zmiennych
boost_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": optimal_boost_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.write("### Ważność zmiennych w modelu boostingu:")
st.dataframe(boost_importances)

# Wizualizacja ważności zmiennych
fig = px.bar(boost_importances, x="Importance", y="Feature", orientation="h", title="Ważność zmiennych w boostingu")
st.plotly_chart(fig)

st.markdown(
    """
        W Boostingu kluczowe znaczenie miały zmienne związane z doświadczeniem pasażera. **`Inflight.entertainment`** była najbardziej wpływową cechą oraz **`Seat.comfort`** i **`On.board.service`** również miały wysoki wpływ, co wskazuje na znaczenie komfortu i obsługi w podróży. Analiza ważności zmiennych podkreśla, że doświadczenie pasażera jest kluczowe dla tego problemu.
    """
)


# Zestawienie wyników wszystkich metod
comparison_results = pd.DataFrame({
    "Model": ["Drzewo Decyzyjne", "Bagging", "Las Losowy", "Boosting"],
    "Accuracy": [dt_metrics["Accuracy"], bagging_metrics["Accuracy"], rf_metrics["Accuracy"], best_boost["Accuracy"]],
    "Precision": [dt_metrics["Precision"], bagging_metrics["Precision"], rf_metrics["Precision"], best_boost["Precision"]],
    "Recall": [dt_metrics["Recall"], bagging_metrics["Recall"], rf_metrics["Recall"], best_boost["Recall"]]
})

st.divider()

# Wyświetlenie tabeli porównawczej
st.write("### Tabela porównawcza wyników różnych metod:")
st.dataframe(comparison_results.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}))

# Wizualizacja porównania wyników
fig = px.bar(
    comparison_results.melt(id_vars="Model", var_name="Metric", value_name="Value"),
    x="Metric", y="Value", color="Model", barmode="group",
    title="Porównanie wyników różnych metod",
    labels={"Value": "Wartość Metryki", "Metric": "Metryka"}
)
st.plotly_chart(fig)

st.write("### Porównanie metod modelowania")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Drzewo Decyzyjne")
    st.write(
        "Modele oparte na pojedynczym drzewie decyzyjnym charakteryzują się niską złożonością "
        "obliczeniową i wysoką interpretowalnością. Jednak metryki jakości (takie jak dokładność "
        "czy czułość) są niższe niż w przypadku metod zespołowych. Drzewa decyzyjne są wrażliwe "
        "na wariancję danych i wykazują tendencję do przeuczenia. Taki model może być dobry jako "
        "wstępny model bazowy."
    )

with col2:
    st.markdown("### Bagging")
    st.write(
        "Bagging zapewnia redukcję wariancji poprzez agregację wielu niezależnie trenowanych drzew "
        "na losowych podzbiorach danych. Metoda zwiększa stabilność i dokładność w porównaniu z "
        "pojedynczym drzewem, jednak nie wprowadza dodatkowej losowości w doborze cech, co może "
        "ograniczać zdolność do dalszej poprawy generalizacji. Bagging jest skuteczny w "
        "podwyższaniu jakości klasyfikacji, lecz jego przewaga nad prostymi drzewami jest głównie "
        "ilościowa, a nie jakościowa."
    )

col3, col4 = st.columns(2)
with col3:
    st.markdown("### Las Losowy (Random Forest)")
    st.write(
        "Las Losowy wykorzystuje losowy wybór cech w każdym węźle, co zwiększa różnorodność drzew "
        "i poprawia generalizację względem Baggingu. Empiryczne wyniki wykazują wyższą precyzję i "
        "stabilniejszą dokładność niż w przypadku pojedynczych drzew i Baggingu. Wadą może być "
        "nieznaczny spadek czułości, wskazujący na zwiększoną liczbę fałszywie negatywnych "
        "wyników. Las losowy dobrze sprawdza się, gdy istotne jest minimalizowanie fałszywych "
        "alarmów (wysoka precyzja), przy jednoczesnym zachowaniu relatywnie dobrej zdolności "
        "generalizacji."
    )

with col4:
    st.markdown("### Boosting")
    st.write(
        "Boosting okazał się najlepszą metodą pod względem Accuracy i Recall. Mechanizm "
        "iteracyjnego dopasowywania słabo działających modeli sprawia, że Boosting jest bardzo "
        "skuteczny w klasyfikacji trudniejszych przypadków. Wyniki wskazują na najwyższy poziom "
        "Recall, co czyni Boosting szczególnie użytecznym w sytuacjach, gdzie minimalizacja "
        "błędów typu II (fałszywie negatywnych) jest kluczowa. Wadą Boostingu jest większa "
        "złożoność obliczeniowa, podatność na nadmierne dopasowanie przy niewłaściwym doborze "
        "parametrów (np. learning rate, liczba estymatorów) oraz potencjalne trudności z "
        "interpretowalnością końcowego modelu. Przy odpowiedniej regularyzacji i doborze "
        "parametrów, Boosting może dawać najwyższą efektywność klasyfikacji spośród "
        "rozważanych metod."
    )
