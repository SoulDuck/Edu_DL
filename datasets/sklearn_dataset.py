from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Boston():
    """
    Boston 집값 예측 관련 데이터셋
    - 506개의 Case
    - 13개의 Feature (crime rate, non-retail business ratio and so on)
    - 1개의 Target (집값)
    """

    def __init__(self):
        """
        Load the boston Dataset

        """
        boston = datasets.load_boston()
        columns = boston.feature_names
        self.boston_df = pd.DataFrame(boston.data, columns=columns)
        self.boston_df.loc[:, "MEDV"] = boston.target

    def __getitem__(self, slc):
        if isinstance(slc, str):
            return self.boston_df.loc[:, slc].values
        elif isinstance(slc, list):
            if isinstance(slc[0], str):
                return self.boston_df.loc[:, slc].values
        elif isinstance(slc, slice):
            return self.boston_df.loc[:, slc].values
        else:
            raise KeyError("적절치 못한 인자가 들어왔습니다")

    def __len__(self):
        return len(self.boston_df)

    @property
    def features(self):
        return self.boston_df.columns[:-1]

    #######
    # Data Summary Helper Method
    #######

    def summary(self):
        return self.boston_df.describe()

    @staticmethod
    def description():
        print(
            "이 데이터셋은 보스턴 지역 내 구역별 집값과 집값에 관련된 특징들을 추출한 데이터셋으로\n"
            "총 506개의 케이스가 존재한다.\n"
            "데이터는\n"
            "총 13개의 feature와\n"
            "1 개의 target(MEDV)로 구성되어있다.\n"
            "\nfeature list ----\n\n"
            "CRIM - 범죄 발생율\n"
            "ZN - 25,000 평방피트 초과 주거지 비율\n"
            "INDUS - 비소매업 면적 비율.\n"
            "CHAS - 찰스강 근처 인지 유무 \n"
            "NOX - 대기 중 NOx 비율\n"
            "RM - 주거지의 평균 방 갯수\n"
            "AGE - 1940년 이전 건축된 주택의 비율\n"
            "DIS - 근무지와의 접근 용이성\n"
            "RAD - 고속도로 접근 용이성\n"
            "TAX - 재산세 비율($10,000)\n"
            "PTRATIO - 학생-선생 비율\n"
            "B - 지역 내 흑인 비율 (1000(Bk - 0.63)^2)\n"
            "LSTAT - 인구 중 하위 계층 비율\n"
            "MEDV - 자가 주택 가격 중간값(단위 : $1000)\n"
        )

    #######
    # Data Visualization Helper Method
    #######

    def scatter(self, x, y, plot_return=False):
        """
        두 Feature 간 산포도

        :param x: x축에 들어가는 Feature 이름
        :param y: y축에 들어가는 Feature 이름
        :param plot_return: plot 객체를 반환할지 유무
        :return:
            plot_return이 True이면,
            plot 객체 반환
        """
        plot = sns.scatterplot(self.boston_df[x],
                               self.boston_df[y])
        plot.set_title("{} - {}".format(x, y))
        if plot_return:
            return plot
        else:
            plt.show()

    def hist(self, x, plot_return=False):
        """
        Feature의 히스토그램

        :param x: Feature 이름
        :param plot_return: plot 객체를 반환할지 유무
        :return:
            plot_return이 True이면,
            plot 객체 반환
        """

        plot = sns.distplot(self.boston_df[x])
        plot.set_title(x)
        if plot_return:
            return plot
        else:
            plt.show()


