import pandas
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

APP_DATE = 'Recommended Application date'
BBCH = 'Requested BBCH Code'
INTERCEPT = 'Crop Interception(%)'
data = pandas.read_csv(Path(__file__).parent / "BBCHGW.csv")
data[APP_DATE] = pandas.to_datetime(data[APP_DATE], dayfirst=True)
print(data.info())
groups = data.drop_duplicates(['Location', 'Crop'])
plot_path = Path(__file__).parent / 'plots'
plot_path.mkdir(exist_ok=True, parents=True)
month_format = mdates.DateFormatter('%b')
for index, group in groups.iterrows():
    mapping = data[(data['Location'] == group['Location']) & (data['Crop'] == group['Crop'])].drop(columns=['Location', 'Crop'])
    name = f"{group['Crop']} {group['Location']}".title()
    group_labels = pandas.date_range(start=mapping[APP_DATE].min(), end=mapping[APP_DATE].max(), freq='MS')
    ax = mapping.plot(x=APP_DATE, y=[BBCH, INTERCEPT], xticks=group_labels)
    ax.xaxis.set_major_formatter(month_format)
    plt.ylim(0,100)
    plt.title(name, fontsize=20)
    plt.savefig(plot_path / f"{name}.svg", bbox_inches='tight')
    plt.close('all')
    print(index, name)