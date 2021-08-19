import matplotlib.pyplot as plt

from crn import Species

class Simulation:
    """
    A completed simulation of a CRN. Contains the time-series information
    of every species in the CRN throughout the simulation. Allows for quick
    plotting and extraction of simulation data.
    완료된 CRN 시뮬레이션. 시뮬레이션 중에 CRN에 있는 모든 species의 시계열 정보를 포함함.
    시뮬레이션 데이터를 빠르게 표시하고 추출할 수 있음.

    This class probably won't be constructed by a user, thus it's
    implementation is more internal.
    이 클래스는 사용자가 구성하지 않으므로 구현이 더 내부적임.

    args:
        sim: Dict[crn.Species, np.ndarray]
            A dictionary of species name to concentration time series.
            This dictionary also contains other fields such as "time" and
            "nothing" which are used for plotting data. This dictionary
            makes no guarantees that these are the only things it will
            contain, it may have new fields that are added if they are
            needed.
            concetnration 시계열과 species name의 dict.
            plotting data를 위해 사용되는 "time"과 "nothing"같은 다른 필드도 포함되어 있음.
            이 dict는 필요한 경우 새 필드가 추가될 수 있음(위의 항목만 포함하는 것은 X)
    """
    def __init__(self, sim, stochastic=False):
        self.sim = sim
        self.stochastic = stochastic
        self.time = sim["time"]
        self.reactions = sim.get("reactions", None)

        del sim["time"]
        del sim["reactions"]

    def __getitem__(self, s):
        if type(s) is not Species:
            raise ValueError(
                "Simulation.__getitem__: tried to get item of non-species. "
                "Type of key must be Species. The type of the key "
                f"passed was {type(s)}")

        return self.sim[s]

    def plot(self, filename=None, title=None):
        """
        Plots the concentration of all of the species over time.
        시간의 경과에 따른 모든 species의 concentration을 표시

        args:
            filename: Optional[str]
                if present, save the plot to a file `filename`. Otherwise,
                the plot will show up as a new window.
                만약 존재하는 경우, 플롯을 파일 'filename'으로 저장함.
                그렇지 않으면 플롯을 새 창으로 표시

            title: Optional[str]
                if present, the plot will have a title `title`.
                존재하는 경우 plot은 'title'이란 타이틀을 가짐.
        """
        if filename:
            backend = plt.get_backend()
            plt.switch_backend("Svg")

        for species, series in sorted(self.sim.items()):
            series = self.sim[species]
            if species.name != "nothing":
                plt.plot(self.time, series, label=f"[{species}]")

        plt.xlabel("time (seconds)")
        if self.stochastic:
            plt.ylabel("molecule counts")
        else:
            plt.ylabel("concentration (M)")

        plt.legend(loc="best")
        if title:
            plt.title(title)

        if filename:
            plt.savefig(filename)
        else:
            plt.show(block=True)

        if filename:
            plt.switch_backend(backend)


