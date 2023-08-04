import numpy as np
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def get_trailer_colors(size, type="matplotlib"):
    set3 = plt.get_cmap('Set3')
    colors = [set3(i) for i in range(size+1)]
    if size > 1:
        colors[1] = set3(5)  # Replace the 2nd color with the 6th one

    if type == "plotly":
        colors = px.colors.qualitative.Set3[: size+1]
        if size > 1:
            colors[1] = px.colors.qualitative.Set3[5]  # Replace the 2nd color with the 6th one

    return colors


def plot_timeline(data: np.array = [], example_name="example") -> None:
    assert isinstance(data, np.ndarray) or (
        len(data) > 0 and len(data[0]) > 0
    ), "Wrong data type"

    df = pd.DataFrame(
        columns=["Truck", "StartTime", "FinishTime", "StartNode", "EndNode", "Trailer"]
    )

    trailer_names = []
    for start_node, target_node, truck, trailer, time in data:
        if trailer == -1:
            trailer = "Trailer {}".format(None)
        else:
            trailer = "Trailer {}".format(trailer)

        if truck == -1:
            truck = None
        else:
            truck = int(truck)

        if trailer not in trailer_names:
            trailer_names.append(trailer)

        trailer_names = sorted(trailer_names, key=lambda x: int(x.split()[1]) if x is not None and x.split()[1]!='None' else float('inf'))

        block = dict(
            Truck=truck,
            StartTime=int(time),
            FinishTime=int(time + 1),
            StartNode="Node {}".format(start_node),
            EndNode="Node {}".format(target_node),
            Trailer=trailer,
        )
        df = df.append(block, ignore_index=True)

    fig = go.Figure()

    # Generate a categorical color scale based on the unique trailer names
    trailer_colors = get_trailer_colors(len(trailer_names), type="plotly")
    y_data = []
    for truck, truck_df in df.groupby("Truck"):
        for idx, row in truck_df.iterrows():
            truck_name = "Truck {} ".format(truck)
            if truck_name not in y_data:
                y_data.append(truck_name)
            fig.add_trace(
                go.Scatter(
                    x=[row["StartTime"], row["FinishTime"]],
                    y=[truck, truck],
                    mode="lines",
                    name=truck_name,
                    line=dict(
                        width=20,
                        color=trailer_colors[trailer_names.index(row["Trailer"])],
                    ),
                    hoverinfo="none",
                    showlegend=False,
                )
            )
            fig.add_annotation(
                x=row["StartTime"],
                y=truck-0.1,
                xref="x",
                yref="y",
                showarrow=False,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=1.5,
                ax=-5,
                ay=0,
                text=row["StartNode"],
                font=dict(size=8),
                align="center",
                valign="bottom",
            )

            fig.add_annotation(
                x=row["FinishTime"],
                y=truck-0.1,
                xref="x",
                yref="y",
                showarrow=False,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=1.5,
                ax=-5,
                ay=0,
                text=row["EndNode"],
                font=dict(size=8),
                align="center",
                valign="bottom",
            )

    # Add legend for the trailer colors
    legend_data = []
    for i, trailer_name in enumerate(trailer_names):
        legend_data.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=trailer_colors[i]),
                name=trailer_name,
                legendgroup=trailer_name,
            )
        )

    base_size = 100
    fig.add_traces(legend_data)
    fig.update_layout(
        # xaxis=dict(range=[df["StartTime"].min() - 0.5, df["FinishTime"].max() + 0.5]), 
        yaxis=dict(
            type='category',
            tickvals=df["Truck"].unique(),
            ticktext=y_data),
        margin=dict(pad=5),
        legend=dict(traceorder="normal"), showlegend=True, width=base_size * len(df), height=1.25*base_size * (len(df["Truck"].unique())+1))
    fig.update_xaxes(title="Steps")
    fig.update_yaxes(autorange="reversed")  # Tasks listed from top to bottom
    fig.show()
    fig.write_image(f"../images/evrp/plot_timeline_{example_name}.png", format="png", scale=300/ 25.4)


if __name__ == "__main__":
    data = torch.tensor(
        [
            [3.0, 2.0, 0.0, -1.0, 0.0],
            [3.0, 1.0, 1.0, -1.0, 1.0],
            [2.0, 1.0, 0.0, 0.0, 2.0],
            [1.0, 2.0, 1.0, 1.0, 3.0],
            [1.0, 2.0, 0.0, 2.0, 4.0],
            [2.0, 2.0, 1.0, -1.0, 5.0],
            [2.0, 2.0, 0.0, -1.0, 6.0],
            [2.0, 2.0, 1.0, -1.0, 7.0],
            [2.0, 2.0, 0.0, -1.0, 8.0],
        ]
    )
    data = data.numpy().astype(int)
    plot_timeline(data)
