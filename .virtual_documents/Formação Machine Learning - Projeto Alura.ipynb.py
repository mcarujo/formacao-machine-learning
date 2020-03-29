get_ipython().run_line_magic("load_ext", " lab_black")


import warnings

warnings.filterwarnings("ignore")


get_ipython().getoutput("pip install pandas=="0.24.0" --quiet")
get_ipython().getoutput("pip install seaborn=="0.9.0" --quiet")
get_ipython().getoutput("pip install scipy=="1.2.0" --quiet")
get_ipython().getoutput("pip install yellowbrick=="0.9.0" --quiet")
get_ipython().getoutput("pip install numpy=="1.16.0" --quiet")


import pandas as pd
import seaborn as sns
import scipy
import yellowbrick
import numpy as np
import matplotlib.pyplot as plt

print("Usando pandas get_ipython().run_line_magic("s"", " % pd.__version__)")
print("Usando seaborn get_ipython().run_line_magic("s"", " % sns.__version__)")
print("Usando scipy get_ipython().run_line_magic("s"", " % scipy.__version__)")
print("Usando yellowbrick get_ipython().run_line_magic("s"", " % yellowbrick.__version__)")
print("Usando numpy get_ipython().run_line_magic("s"", " % np.__version__)")


# Vamos configurar o pandas para usar impressão de ponto flutuante com 3 casas decimais
pd.set_option("display.float_format", lambda x: "get_ipython().run_line_magic(".3f"", " % x)")


# solução
usecols = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_COMP1",
    "NU_NOTA_COMP2",
    "NU_NOTA_COMP3",
    "NU_NOTA_COMP4",
    "NU_NOTA_COMP5",
]
enem = pd.read_csv(
    "input/MICRODADOS_ENEM_2017.csv", usecols=usecols, sep=";", encoding="iso-8859-1"
)


enem.head()


print("get_ipython().run_line_magic("d", " elementos e %d colunas\" % (enem.shape[0], enem.shape[1]))")
if enem.shape[0] get_ipython().getoutput("= 6731341:")
    print("ERRO! No conjunto de 2017 existem 6731341 dados")
if enem.shape[1] get_ipython().getoutput("= 9:")
    print("ERRO! Carregue somente 9 colunas relativas as notas")


# solução
todas_as_notas = [
    #     "TP_STATUS_REDACAO",
    "NU_NOTA_COMP1",
    "NU_NOTA_COMP2",
    "NU_NOTA_COMP3",
    "NU_NOTA_COMP4",
    "NU_NOTA_COMP5",
    #     "NU_NOTA_REDACAO",
]


enem[todas_as_notas].head()


# solução
enem.dropna(inplace=True)
enem.NU_NOTA_MT.head()


# solução e impressão
enem["nota_total"] = 0
enem["nota_total"] = enem.sum(axis=1)


# solução histograma e descrição
print(enem.nota_total.describe())
sns.distplot(enem.nota_total)
plt.title("Distribuição de notas (final) do Enem")
plt.ylabel("Quatidade normalizada")
plt.xlabel("Nota")


import matplotlib.pyplot as plt

# solução sua função de sampling


def eda_sample(enem):
    # seed
    # 1% de sample em enem_eda
    enem_eda = enem.sample(frac=0.01, random_state=745)
    print("Enem EDA sampling tem a distribuição")
    # descreva a nota_total
    enem_eda.nota_total.describe()
    # plote o histograma da nota_total e mostre com plt.show()
    sns.distplot(enem_eda.nota_total)
    plt.title("Enem EDA sampling tem a distribuição")
    plt.ylabel("Quatidade normalizada")
    plt.xlabel("Nota")
    plt.show()
    return enem_eda


enem_eda = eda_sample(enem)


# solução
sns.set(style="white")

# Compute the correlation matrix
correlacoes = enem.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(correlacoes, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    correlacoes,
    mask=mask,
    cmap="Reds",
    annot=True,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)
plt.title("Correlação das notas parciais com a nota final.")


if correlacoes.shape get_ipython().getoutput("= (10, 10):")
    print(
        "A matriz de correlação deveria ser entre 10 notas, totalizando 10 linhas por 10 colunas"
    )


# solução: cálculo da tabela de correlação com todas as notas
correlacao_com_nota_total = correlacoes.corr()


print(correlacao_com_nota_total)


# solução gráfico
def plota_correlacao(dados):
    sns.set(style="white")
    plt.figure(figsize=(12, 6))
    corr = dados.corr()["nota_total"].sort_values()
    sns.barplot(y=corr.index, x=corr.values)


plota_correlacao(correlacao_com_nota_total)


# solução: definindo interesse e imprimindo os 5 primeiros elementos
interesse = enem_eda[["NU_NOTA_MT", "NU_NOTA_LC", "nota_total"]]


# Solução: a função de split

from sklearn.model_selection import train_test_split


def split(dados):
    # seed
    # train_test_split
    train_x, test_x, train_y, test_y = train_test_split(
        dados.drop("nota_total", axis=1),
        dados["nota_total"],
        test_size=0.2,
        random_state=42367,
    )
    print("*" * 80)
    print(
        "Quebrando em treino (x,y) e teste (x,y)",
        train_x.shape,
        train_y.shape,
        test_x.shape,
        test_y.shape,
    )
    print("Usando colunas get_ipython().run_line_magic("s", " como X\" % str(train_x.columns.values))")
    print("Desvio padrão do conjunto de testes", test_y.std())
    return train_x, test_x, train_y, test_y


# Código pronto

train_x, test_x, train_y, test_y = split(interesse)
if train_x.shape[1] get_ipython().getoutput("= 2:")
    print("*" * 80)
    print("Erro! Você deveria possuir somente duas colunas em X")
    print("*" * 80)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# Solução: função para executar a regressão linear


def roda_regressao_linear(train_x, test_x, train_y, test_y):

    # crie o modelo, treine com os dados de treino
    # calcule o r2_score com os dados de teste
    # calcule a predição e os dois tipos de erros
    model = LinearRegression()
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)

    mse = mean_squared_error(test_y, test_pred)
    mae = mean_absolute_error(test_y, test_pred)
    r2 = r2_score(test_y, test_pred)

    print("*" * 80)
    print("r2 score", r2)
    print("mse", mse)
    print("mae", mae)

    return model


# código pronto
roda_regressao_linear(train_x, test_x, train_y, test_y)


# código pronto

from yellowbrick.regressor import PredictionError


def visualiza_erros(train_x, train_y, test_x, test_y):
    visualizer = PredictionError(LinearRegression())
    visualizer.fit(train_x, train_y)
    visualizer.score(test_x, test_y)
    visualizer.poof()


visualiza_erros(train_x, train_y, test_x, test_y)


# código pronto

from yellowbrick.regressor import ResidualsPlot


def visualiza_erros(train_x, train_y, test_x, test_y):
    visualizer = PredictionError(LinearRegression())
    visualizer.fit(train_x, train_y)
    visualizer.score(test_x, test_y)
    visualizer.poof()

    visualizer = ResidualsPlot(LinearRegression())
    visualizer.fit(train_x, train_y)
    visualizer.score(test_x, test_y)
    visualizer.poof()


visualiza_erros(train_x, train_y, test_x, test_y)


# código pronto
def regressao_completa_para(notas):
    interesse = enem_eda[notas]
    train_x, test_x, train_y, test_y = split(interesse)
    model = roda_regressao_linear(train_x, test_x, train_y, test_y)
    visualiza_erros(train_x, train_y, test_x, test_y)


# solução 1: teste com todas as notas
regressao_completa_para(
    [
        "NU_NOTA_CN",
        "NU_NOTA_CH",
        "NU_NOTA_LC",
        "NU_NOTA_MT",
        "NU_NOTA_COMP1",
        "NU_NOTA_COMP2",
        "NU_NOTA_COMP3",
        "NU_NOTA_COMP4",
        "NU_NOTA_COMP5",
        "nota_total",
    ]
)


# solução 2: teste outra combinação
regressao_completa_para(
    [
        "NU_NOTA_COMP1",
        "NU_NOTA_COMP2",
        "NU_NOTA_COMP3",
        "NU_NOTA_COMP4",
        "NU_NOTA_COMP5",
        "nota_total",
    ]
)


# solução 3: teste outra combinação
regressao_completa_para(
    ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "nota_total"]
)


# solução 4: teste outra combinação
regressao_completa_para(
    ["NU_NOTA_CN", "NU_NOTA_LC", "nota_total",]
)


# solução 5: teste outra combinação
regressao_completa_para(["NU_NOTA_CN", "NU_NOTA_CH", "nota_total"])


# solução 6: teste outra combinação
regressao_completa_para(["NU_NOTA_CN", "NU_NOTA_MT", "nota_total"])


# solução 7: teste outra combinação
regressao_completa_para(["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "nota_total"])


# solução
regressao_completa_para(["NU_NOTA_LC", "NU_NOTA_COMP3", "NU_NOTA_MT", "nota_total"])


# solução

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def gera_regressores():
    # gere os modelos em uma lista
    modelos = [
        LinearRegression(),
        Lasso(),
        Ridge(),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
    ]
    return modelos


# teste

if len(gera_regressores()) get_ipython().getoutput("= 5:")
    print("Erroget_ipython().getoutput("!! São 5 regressores que queremos testar!")")


# solução
def escolhe_dados(dados, colunas):
    train_x, test_x, train_y, test_y = split(dados[colunas])
    sns.distplot(train_y)
    plt.show()

    return train_x, test_x, train_y, test_y


# solução:

import time


def treina_e_mede_regressor(modelo, train_x, test_x, train_y, test_y):
    tic = time.time()
    modelo.fit(train_x, train_y)
    tac = time.time()
    tempo_de_treino = tac - tic

    test_pred = modelo.predict(test_x)
    mse = mean_squared_error(test_y, test_pred)
    mae = mean_absolute_error(test_y, test_pred)

    print("Resultado", modelo, mse, mae)

    return mse, mae, tempo_de_treino


# código pronto

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

scaler = StandardScaler()


def analisa_regressao(dados):
    train_x, test_x, train_y, test_y = escolhe_dados(
        dados, ["NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_COMP3", "nota_total"]
    )

    scaler.fit(train_x)
    train_x = scaler.transform(train_x)

    resultados = []
    for modelo in gera_regressores():

        pipe = make_pipeline(StandardScaler(), modelo)

        mse, mae, tempo_de_treino = treina_e_mede_regressor(
            pipe, train_x, test_x, train_y, test_y
        )

        resultados.append([modelo, pipe, tempo_de_treino, mse, mae])

    resultados = pd.DataFrame(
        resultados, columns=["modelo", "pipe", "tempo_de_treino", "mse", "mae"]
    )
    return test_x, test_y, resultados


test_x, test_y, notas = analisa_regressao(enem_eda)
notas[["modelo", "mse", "mae", "tempo_de_treino"]]


# solução: histograma
plt.xlabel("Nota")
plt.ylabel("Quantidade de notas normalizada")
plt.title("Distribuição das notas de test")
sns.distplot(test_y)


# solução


def top_p(serie, p=0.75):
    # calcule o quantil p
    quant = serie.quantile(p)
    print("quantile encontrado", quant)
    # defina y como sendo uma serie de 1s e 0s. 1 se o valor da serie for maior que o quantil, 0 se menor
    y = [1 if value > quant else 0 for value in serie]
    return pd.Series(y)


# teste do top 25%
top_25 = top_p(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), p=0.75).values
if not np.array_equal(top_25, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]):
    print("Não retornou o top 25% corretamente, deveria ser ", top_25)


# teste do top 10%

top_10 = top_p(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), p=0.90).values
if not np.array_equal(top_10, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]):
    print("Não retornou o top 10% corretamente, deveria ser", top_10)


# código pronto

y_top25 = top_p(test_y)
y_top25.mean()


# código pronto

from yellowbrick.target import ClassBalance

visualizer = ClassBalance(labels=["75get_ipython().run_line_magic("",", " \"25%\"])")
visualizer.fit(y_top25)
visualizer.poof()


# código pronto

from yellowbrick.target import BalancedBinningReference

visualizer = BalancedBinningReference()
visualizer.fit(train_y)
visualizer.poof()


# código pronto
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# solução


def gera_classificadores():
    # defina seus modelos
    modelos = [
        DummyClassifier(strategy="most_frequent"),
        LogisticRegression(),
        RidgeClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=10),
        SVC(),
    ]
    return modelos


# código pronto

if len(gera_classificadores()) get_ipython().getoutput("= 6:")
    print("Erroget_ipython().getoutput("!! São 6 classificadores que queremos testar!")")


def split_classificacao(dados):
    # faça o seed do numpy
    np.random.seed(42367)
    # defina X como todas as colunas de `dados` exceto top_p
    X = dados.drop("top_p", axis=1)
    # defina y como somente a coluna top_p
    Y = dados["top_p"]
    # quebre em treino e teste, usando estratificação baseada em y
    train_x, train_y, test_x, test_y = train_test_split(X, Y, stratify=Y)
    print("*" * 80)
    print(
        "Quebrando em treino (x,y) e teste (x,y)",
        train_x.shape,
        train_y.shape,
        test_x.shape,
        test_y.shape,
    )
    print("Usando colunas get_ipython().run_line_magic("s", " como X\" % str(train_x.columns.values))")
    print("Média do conjunto de testes", test_y.mean())
    return train_x, test_x, train_y, test_y


# código pronto: teste

interesse = enem_eda[["nota_total", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_COMP3"]]
interesse["top_p"] = list(top_p(interesse["nota_total"]))
interesse = interesse[["top_p", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_COMP3"]]

train_x, test_x, train_y, test_y = split_classificacao(interesse)

if train_x.shape[1] get_ipython().getoutput("= 3:")
    print("*" * 80)
    print("Erro! Você deveria possuir somente três colunas em X")
    print("*" * 80)

if test_y.mean() <= 0.24 or test_y.mean() >= 0.26:
    print("*" * 80)
    print(
        "Erro! Você deveria capturar somente o top 25% e usar estratificação no split"
    )
    print("*" * 80)


# solução:

import time


def treina_e_mede_classificador(pipe, nome, train_x, test_x, train_y, test_y):
    tic = time.time()
    pipe.fit(train_x, train_y)
    tac = time.time()
    tempo_de_treino = tac - tic

    # calcule a accuracy_score
    accuracy_score = pipe.score(test_x, test_y)
    print("Resultado", nome, accuracy_score)

    return accuracy_score, tempo_de_treino


# solução:


def escolhe_dados_para_classificacao(dados, colunas, p):
    interesse = dados[colunas].drop("nota_total", axis=1)
    nota_total = dados["nota_total"]
    interesse["top_p"] = list(top_p(nota_total, p))

    colunas.remove("nota_total")
    interesse = interesse[[*colunas, "top_p"]]

    train_x, train_y, test_x, test_y = split_classificacao(interesse)
    train_y.hist()
    plt.show()
    return train_x, test_x, train_y, test_y


# testando a escolha

train_x, test_x, train_y, test_y = escolhe_dados_para_classificacao(
    enem_eda, ["nota_total", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_COMP3"], p=0.75
)

if train_x.shape[1] get_ipython().getoutput("= 3:")
    print("*" * 80)
    print("Erro! Você deveria possuir somente três colunas em X")
    print("*" * 80)

if test_y.mean() <= 0.24 or test_y.mean() >= 0.26:
    print("*" * 80)
    print(
        "Erro! Você deveria capturar somente o top 25% e usar estratificação no split"
    )
    print("*" * 80)


# código pronto


def analisa_classificacao(dados, p=0.75):

    colunas = ["nota_total", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_COMP3"]
    train_x, test_x, train_y, test_y = escolhe_dados_para_classificacao(
        dados, colunas, p=p
    )

    resultados = []
    for modelo in gera_classificadores():
        nome = type(modelo).__name__
        pipe = make_pipeline(StandardScaler(), modelo)
        accuracy_score, tempo_de_treino = treina_e_mede_classificador(
            pipe, nome, train_x, test_x, train_y, test_y
        )
        resultados.append([nome, modelo, pipe, tempo_de_treino, accuracy_score])

    resultados = pd.DataFrame(
        resultados,
        columns=["tipo", "modelo", "pipe", "tempo_de_treino", "accuracy_score"],
    )
    return test_x, test_y, resultados.set_index("tipo")


# solução top 25%
notas = analisa_classificacao(enem_eda)[2]
# rode a analisa_classificacao e armazene test_x, test_y e notas
notas[["accuracy_score", "tempo_de_treino"]]


# solução top 20%
notas = analisa_classificacao(enem_eda, 0.2)[2]
# rode a analisa_classificacao e armazene test_x, test_y e notas
notas[["accuracy_score", "tempo_de_treino"]]


# solução top 10%
notas = analisa_classificacao(enem_eda, 0.1)[2]
# rode a analisa_classificacao e armazene test_x, test_y e notas
notas[["accuracy_score", "tempo_de_treino"]]


# solução top 5%
notas = analisa_classificacao(enem_eda, 0.05)[2]
# rode a analisa_classificacao e armazene test_x, test_y e notas
notas[["accuracy_score", "tempo_de_treino"]]


# solução top 1%
notas = analisa_classificacao(enem_eda, 0.01)[2]
# rode a analisa_classificacao e armazene test_x, test_y e notas
notas[["accuracy_score", "tempo_de_treino"]]


# solução bottom 25%
notas = analisa_classificacao(enem_eda, 0.25)[2]
# rode a analisa_classificacao e armazene test_x, test_y e notas
notas[["accuracy_score", "tempo_de_treino"]]


# código pronto: rodando para top 25%
test_x, test_y, notas = analisa_classificacao(enem_eda, 0.25)
# rode a analisa_classificacao e armazene test_x, test_y e notas de 25%
notas[["accuracy_score", "tempo_de_treino"]]


# código pronto

import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm):

    classes = ["Não topo 25get_ipython().run_line_magic("",", " \"Topo 25%\"]")

    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matriz de confusão normalizada")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], ".2f") + "get_ipython().run_line_magic("",", "")
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("Classe real")
    plt.xlabel("Classe predita")
    plt.tight_layout()
    plt.show()


def print_confusion_for(test_x, test_y, model):
    pred_y = model.predict(test_x)
    print("Acurácia do modelo em teste", model.score(test_x, test_y))
    cnf_matrix = confusion_matrix(test_y, pred_y)

    plot_confusion_matrix(cnf_matrix)


# código pronto

print_confusion_for(test_x, test_y, notas.loc["LogisticRegression"]["pipe"])


print_confusion_for(test_x, test_y, notas.loc["DummyClassifier"]["pipe"])


# código pronto

print_confusion_for(test_x, test_y, notas.loc["SVC"]["pipe"])


# código pronto: separando os dados não usados para treino (que foram os usados em eda)

usados_no_eda = enem_eda.index
a_usar = ~enem.index.isin(usados_no_eda)
enem_validacao = enem[a_usar]
print("Para otimização temos get_ipython().run_line_magic("d", " elementos\" % len(enem_validacao))")
del a_usar
del usados_no_eda


def separa_dados_de_classificacao_para_validacao(dados):
    X = dados[["NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_COMP3"]]
    y = pd.Series(list(top_p(dados["nota_total"])))
    print("Média da validação", y.mean())
    return X, y


# solução

from sklearn.model_selection import cross_val_score


def treina_e_valida_modelo_de_classificacao(dados, modelo):
    # calcule X e y usando a função anterior
    X, y = separa_dados_de_classificacao_para_validacao(dados)
    scores = cross_val_score(modelo, X, y, cv=5)
    mean = scores.mean()
    std = scores.std()
    print("Acurácia entre [get_ipython().run_line_magic(".2f,%.2f]"", " % (100 * mean - 2 * std, 100 * mean + 2 * std))")

    modelo.fit(X, y)
    print_confusion_for(X, y, modelo)


pipeline_logistica = make_pipeline(StandardScaler(), LogisticRegression())
treina_e_valida_modelo_de_classificacao(enem_validacao, pipeline_logistica)


# solução: implemente o código que falta
from sklearn.metrics import accuracy_score


class HeuristicaTop25:
    def fit(self, X, y=None):
        if X.shape[1] get_ipython().getoutput("= 3:")
            print("Erroget_ipython().getoutput("!! Estávamos esperando 3 colunas!")")
        parcial = X.sum(axis=1)
        self.top_25_quantile = pd.Series(parcial).quantile(0.75)
        print("top 25 quantile é get_ipython().run_line_magic(".2f"", " % self.top_25_quantile)")
        return self

    def predict(self, X, y=None):
        parcial = X.sum(axis=1)
        y_pred_true_false = [
            True if value >= self.top_25_quantile else False for value in parcial
        ]  # compare a soma parcial com o self.top25_quantile
        y_pred = [
            1 if value else 0 for value in y_pred_true_false
        ]  # 1 se for maior ou igual, 0 caso contrário
        return y_pred

    def score(self, X, y=None):
        return accuracy_score(y, self.predict(X, y))

    def get_params(self, deep=True):
        return {}


treina_e_valida_modelo_de_classificacao(enem_validacao, HeuristicaTop25())
