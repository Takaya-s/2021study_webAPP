import hashlib
import os
import sqlite3
from pathlib import Path


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
import streamlit as st
from plotly.subplots import make_subplots

py.init_notebook_mode(connected=True)

# Security
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


# DB Management
USER_DB = ".userdata.db"
conn = sqlite3.connect(USER_DB)
c = conn.cursor()


def cache_state(state):
    return state


# DB  Functions
def create_usertable():
    c.execute("CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)")


def add_userdata(username, password):
    c.execute(
        "INSERT INTO userstable(username,password) VALUES (?,?)", (username, password)
    )
    conn.commit()


@st.cache
def login_user(username, password):
    c.execute(
        "SELECT * FROM userstable WHERE username =? AND password = ?",
        (username, password),
    )
    data = c.fetchall()
    return data


def view_all_users():
    c.execute("SELECT * FROM userstable")
    data = c.fetchall()
    return data


########################################################################


@st.cache(allow_output_mutation=True)
def load_data(homedir: Path):
    blood_path = homedir.joinpath("blood_random")
    screening_path = homedir.joinpath("screening_random")

    lsi_list = [
        pd.read_csv(p, sep="\t", encoding="utf16")
        for p in blood_path.glob("*.csv")
        if not p.name.startswith(".") and p.is_file()
    ]
    scr_list = [
        pd.read_csv(p, encoding="utf8")
        for p in screening_path.glob("*.csv")
        if not p.name.startswith(".") and p.is_file()
    ]

    return lsi_list, scr_list


@st.cache(allow_output_mutation=True)
def process_data(scr: pd.DataFrame, lsi: pd.DataFrame):
    scr.replace(
        {
            "HINCOME": {np.NaN: 10},
            "SEX": {1: "male", 2: "female"},
            "MARRIED": {1: "未婚", 2: "既婚"},
            "CHILD": {1: "子供なし", 2: "子供あり"},
        },
        inplace=True,
    )
    # data wrangling
    ageid_dict = {
        3: "20-24",
        4: "25-29",
        5: "30-34",
        6: "35-39",
        7: "40-44",
        8: "45-49",
        9: "50-54",
        10: "55-59",
        11: "60-",
    }
    ageid_dict2 = {
        3: "20's",
        4: "20's",
        5: "30's",
        6: "30's",
        7: "40's",
        8: "40's",
        9: "50's",
        10: "50's",
        11: "60's",
    }
    hincom_dict = {
        1: "less than 2M",
        2: "2-4M",
        3: "4-6M",
        4: "6-8M",
        5: "8-10M",
        6: "10-12M",
        7: "12-15M",
        8: "15-20M",
        9: "more than 20M",
        10: "unknown",
    }
    job_dict = {
        1: "公務員",
        2: "経営者役員",
        3: "会社員事務系",
        4: "会社員技術系",
        5: "会社員その他",
        6: "自営業",
        7: "自由業",
        8: "専業主婦主夫",
        9: "パートアルバイト",
        10: "学生",
        11: "その他",
        12: "無職",
    }
    pref_dict = {11: "埼玉県", 12: "千葉県", 13: "東京都", 14: "神奈川県"}
    latlon = {
        11: {"lat": 35.85694, "lon": 139.64889},  # 埼玉
        12: {"lat": 35.60472, "lon": 140.12333},  # 千葉
        13: {"lat": 35.68944, "lon": 139.69167},  # 東京
        14: {"lat": 35.44778, "lon": 139.6425},
    }  # 神奈川

    renamedict_list = [ageid_dict2, pref_dict, job_dict, hincom_dict]
    renameindex_list = ["AGEID", "PREFECTURE", "JOB", "HINCOME"]
    renamecol_list = ["Age", "Prefecture", "Jobtype", "Hincome"]

    for dic, ind, col in zip(
        renamedict_list, renameindex_list, renamecol_list
    ):  # catch three lists with zip
        # print(dic, ind, col)
        temp_pd = pd.DataFrame.from_dict(
            dic, orient="index"
        ).reset_index()  # pd.DataFrame from dict in for loop
        temp_pd.rename(columns={"index": ind, 0: col}, inplace=True)

        scr = scr.merge(temp_pd, on=ind)
        stress_res_list = ["活気", "イライラ", "疲労感", "不安感", "抑うつ感", "身体愁訴"]
    stress_res_male_item_list = [f"Score{item}" for item in range(7, 13)]
    stress_res_female_item_list = [f"Score{item}" for item in range(13, 19)]

    scr[stress_res_list] = ""
    for res, male, female in zip(
        stress_res_list, stress_res_male_item_list, stress_res_female_item_list
    ):
        scr[res] = np.where(scr.SEX == 1, scr[male], scr[female])

    # st.write(scr)

    scr["Sex"] = pd.Categorical(
        scr.SEX, ordered=True, categories=["male", "female"]  # Categorize sex series.
    )

    scr["Married"] = pd.Categorical(
        scr.MARRIED, ordered=True, categories=["未婚", "既婚"]  # Categorize married series.
    )
    scr["Child"] = pd.Categorical(
        scr.CHILD, ordered=True, categories=["子供なし", "子供あり"]  # Categorize age series.
    )

    scr["Age"] = pd.Categorical(
        scr.Age, ordered=True, categories=["20's", "30's", "40's", "50's", "60's"]
    )  # Categorize age series.
    scr["CELLNAME"] = pd.Categorical(
        scr.CELLNAME, ordered=True, categories=["Low", "Middle", "High"]
    )  # Categorize stress series.

    scr = scr.sort_values("SAMPLEID").reset_index(drop=True)

    data = scr.merge(lsi, left_on="SAMPLEID", right_on="ID")  # Merge two data frame

    lsi_datasheet_list = (
        ["SAMPLEID"]
        + ["Age", "Sex", "Married", "Child", "Jobtype", "Hincome", "CELLNAME"]
        + stress_res_list
        + lsi.columns[3:].values.tolist()
    )
    data = data[lsi_datasheet_list]  # Select columns
    return scr, lsi, data


def init_datapage():
    cwd = Path(os.getcwd())
    st.write(cwd)
    outpath = cwd.joinpath("output")
    outfigpath = outpath.joinpath("Figure")
    # make outputpath
    if not (outpath.exists() and outpath.is_dir()):
        outpath.mkdir(exist_ok=True)
        outfigpath.mkdir(exist_ok=True)

    lsi_list, scr_list = load_data(cwd)

    # preprocess data
    scr, lsi, data = process_data(scr_list[0], lsi_list[0])
    ###########   data merge with LSI blood test #####################
    st.write(data)
    ############### widget #######################
    st.sidebar.header("User Input Features")
    datasheet_ = st.sidebar.selectbox(
        "Select data sheet", ["Screening", "LSI blood test"]
    )
    column_ = st.sidebar.selectbox(
        "Sort column",
        ["Age", "Sex", "Married", "Child", "Jobtype", "Hincome", "CELLNAME"],
    )

    if not datasheet_:
        st.write("## Fisrt, select menu in side bar")

    st.write("---")

    size_sunburst = pd.DataFrame(
        scr.groupby(column_).size().rename("Size").reset_index()
    )
    size_sunburst["parents"] = ""
    size_sunburst.rename(
        columns={"Size": "count", f"{column_}": "labels"}, inplace=True
    )
    if datasheet_ == "Screening":
        screening_dataprocess(scr, column_)

    if datasheet_ == "LSI blood test":
        st.markdown("---")
        st.header("LSI blood test by stress level")
        # lsi_value_item = "Cortisol"
        lsi_value_item = st.selectbox(
            "Select item", options=lsi.columns[3:].values.tolist()
        )
        lsi_data_by_stress(data, lsi_value_item)

        st.markdown("---")
        st.markdown("###  Select the 1st axis")
        grouping = st.checkbox("Grouping")
        if grouping:

            lsi_group_item_1st = st.selectbox(
                "Select a variable you want to group",
                options=[
                    "Age",
                    "Sex",
                    "Married",
                    "Child",
                    "Jobtype",
                    "Hincome",
                    "Prefecture",
                ],
            )

            lsi_by_multiple_variables(
                data, lsi_group_item_1st, lsi_value_item=lsi_value_item
            )
            # try:
            #     lsi_by_multiple_variables(data, lsi_group_item_1st)
            # except:
            #     st.error("Please select the 1st variable")

        else:
            pass


def boxplot_LSI_single(
    data,
    lsi_value_item=None,
):

    if lsi_value_item:
        fig = px.box(data, x="CELLNAME", y=lsi_value_item)
        st.write(fig)
        return st.plotly_chart(fig, use_container_width=True)
    else:
        pass


def summary_LSI_single_by_stress(data, lsi_value_item=None):
    if lsi_value_item:
        summary = (
            data.groupby("CELLNAME")
            .agg(["max", "min", "mean", "median", "std"])[lsi_value_item]
            .reset_index()
        )
        return summary

        # return st.write(summary)
    else:
        pass


############# Hereafter main part ######################


def scr_likert_group(scr, select_key=None):
    stress_res_list = ["活気", "イライラ", "疲労感", "不安感", "抑うつ感", "身体愁訴"]
    scr_l = scr.melt(
        value_vars=stress_res_list,
        var_name="Scale",
        value_name="Score",
        id_vars=[
            "Age",
            "SAMPLEID",
            "Prefecture",
            "SEX",
            "Married",
            "Child",
            "Jobtype",
            "Hincome",
            "CELLNAME",
        ],
    )  # .groupby(["CELLNAME", "活気"]).size().rename("Size").reset_index()
    #
    if not select_key:
        st.write("notselectyyyyyyyyyyyyyyyy")
        scr_l_g = (
            scr_l.groupby(["Scale"])["Score"]
            .apply(lambda x: x.value_counts() / len(x) * 100)
            .reset_index()
            .rename(columns={"level_1": "Score", "Score": "Rate"})
        )  # cal rate

    else:
        st.write("selectwwwwwwwwwwww")
        scr_l_g = (
            scr_l.groupby(["Scale", select_key])["Score"]
            .apply(lambda x: x.value_counts() / len(x) * 100)
            .reset_index()
            .rename(columns={"level_2": "Score", "Score": "Rate"})
        )  # cal rate
        # st.write(scr_l_g)

    scr_l_g["Score"] = pd.Categorical(
        scr_l_g["Score"].astype(str), ordered=True, categories=["1", "2", "3", "4", "5"]
    )
    scr_l_g.sort_values("Score", ascending=True, inplace=True)
    scr_l_g["Rate"] = np.round(scr_l_g["Rate"], 2)

    # st.write(scr_l_g)
    return scr_l_g


def plotly_likert(scr_l_g, select_key=None):

    ## go figure object
    fig = go.Figure()
    xd = []
    yd = []
    colors = [
        "rgba(38, 24, 74, 0.8)",
        "rgba(71, 58, 131, 0.8)",
        "rgba(122, 120, 168, 0.8)",
        "rgba(164, 163, 204, 0.85)",
        "rgba(190, 192, 213, 1)",
    ]

    for i, sco in enumerate(scr_l_g.Score.unique()):
        data = scr_l_g[scr_l_g["Score"] == sco]
        fig.add_trace(
            go.Bar(
                x=data.Rate,
                y=data.Scale,
                orientation="h",
                name=sco,
                marker=dict(color=colors[i]),
                hovertemplate="<i>Scale</i>: %{y}" + "<br><i>Rate</i>: <b>%{x} </b>%",
                # customdata=sc,
            )
        )

    layout = dict(
        legend_title="Score",
        xaxis=dict(
            title="%",
            showgrid=True,
            showline=True,
            showticklabels=True,
            zeroline=True,
            domain=[0.15, 1],
        ),
        yaxis=dict(
            title="Item",
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False,
        ),
        barmode="relative",
        height=400,
        width=700,
        paper_bgcolor="rgb(248, 248, 255)",
        plot_bgcolor="rgb(248, 248, 255)",
        yaxis_autorange="reversed",
        bargap=0.04,
        margin=dict(l=20, r=100, t=140, b=80),
        showlegend=True,
    )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)
    return fig, colors, layout


######### subburst chart ###################


def export_datasheet(
    datasheet, path, filename="datafromstreamlit", current=True, parent=None
):
    if current and parent == None:
        outpath = path / "output"
        outfigpath = outpath / "Figure"

    elif current == None and parent == True:
        outpath = path.parent / "output"
        outfigpath = outpath / "Figure"

    else:
        st.error("please select either the current path or the parent path")

    # make outputpath
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
        os.makedirs(outfigpath)

    return datasheet.to_csv(outpath / f"{filename}.csv")


# st.write(scr.sort_values("PREFECTURE").Prefecture.unique()[1:])
# st.sidebar.multiselect("Age", scr.sort_values("Age").Age.unique())
# st.dataframe(scr[scr["Prefecture"] == ])


def screening_dataprocess(scr, column_):
    st.header("Screening data")
    st.write("You selected a {} variable".format(column_))
    # side = st.sidebar.multiselect("Prefecture", scr.sort_values("PREFECTURE").Prefecture.unique())

    # st.header("Show numbers")
    if st.checkbox("Show a sample size"):

        size_ = pd.DataFrame(scr.groupby(column_).size().rename("Size").reset_index())

        st.write(size_)

    # with st.form("Do you want to see data grouped by variabels?"):

    st.write("---")

    st.markdown("- BJSQ score population")

    scr_l_g = scr_likert_group(scr, select_key=None)

    fig, colors, layout = plotly_likert(scr_l_g)

    st.write("Do you want to see data grouped by variables?")
    group_yes = st.checkbox("YES")

    if group_yes:
        # plot_likert(scr)
        select_val = st.selectbox(
            "Select variables",
            ["", "Age", "Sex", "Married", "Child", "Jobtype", "Hincome", "CELLNAME"],
        )
        st.write(f"{select_val} selected")

        if not select_val == "":
            select_key = select_val
            scr_l_g = scr_likert_group(scr, select_key=select_key)
            scr_l_g.sort_values("Scale", inplace=True)

            if select_key == select_key:
                n = len(scr_l_g[select_key].unique())
                # st.write(n)
                # fig, colors, layout = plotly_likert(scr_l_g)
                fig = make_subplots(
                    rows=n,
                    cols=1,
                    subplot_titles=(scr_l_g[select_key].sort_values().unique()),
                )
                for i, (a, d) in enumerate(scr_l_g.groupby(select_key)):

                    for ii, sco in enumerate(d.Score.sort_values().unique()):
                        # st.write(i, sco)
                        data = d[d["Score"] == sco]
                        # st.write(data)
                        fig.add_trace(
                            go.Bar(
                                x=data.Rate,
                                y=data.Scale,
                                orientation="h",
                                name=sco,
                                marker=dict(color=colors[ii]),
                                hovertemplate="<i>Scale</i>: %{y}"
                                + "<br><i>Rate</i>: <b>%{x} </b>%",
                                # customdata=sc,
                            ),
                            row=i + 1,
                            col=1,
                        )

                fig.for_each_trace(lambda trace: trace.update(showlegend=False))
                fig.update_layout(
                    barmode="relative",
                    height=1000,
                    width=700,
                )
                st.plotly_chart(fig)

            selectexport = st.checkbox("Export grouped data as csv")
            if selectexport:
                exportform = st.form("export csv form")
                with exportform:

                    filename = st.text_input("Type filename you want to save as csv")
                    filename

                    pathselect = st.radio(
                        "Which path do you want to save datasheet as csv in?",
                        ("Current", "Parent"),
                    )
                    if pathselect == "Current":
                        current, parent = (True, None)
                    elif pathselect == "Parent":
                        current, parent = (None, True)

                    submitted = st.form_submit_button("Submit")
                    if submitted:
                        if not filename:
                            st.error("Error: Please enter filename!")

                        else:
                            export_datasheet(
                                scr_l_g,
                                path,
                                filename=filename,
                                current=current,
                                parent=parent,
                            )

            if not select_val:
                st.error("Please select at least one variable.")

    else:
        # plot_likert(scr)
        pass

    ########   Vaccination rate    ###########
    if st.checkbox("Vaccination rate  chart"):
        size_sunburst_vaccine = (
            scr.groupby(["Q6S6", column_])
            .count()
            .SAMPLEID.reset_index()
            .rename(columns={"Q6S6": "labels", "SAMPLEID": "count", column_: "parents"})
        )
        # st.write(size_sunburst)
        # st.write(size_sunburst_vaccine)

        df = size_sunburst_vaccine.copy()
        df["parents"] = df["parents"].astype(str)
        df["labels"] = df["labels"].astype(str)
        df.reset_index(inplace=True, drop=True)
        df["vaccine"] = f"Vaccination<br>by {column_}"

        fig = px.sunburst(
            df,
            path=["vaccine", "parents", "labels"],
            values="count",
            branchvalues="total"
            # color_discrete_map={},#,
            # color = "count"
        )

        layout = dict(title=f"Vaccination rate by {column_}", width=500, height=500)
        fig.update_layout(layout)
        st.plotly_chart(fig, use_container_width=True)

    # 都道府県別参加者

    if st.checkbox("都道府県別参加者"):

        participate = pd.merge(
            scr.groupby("PREFECTURE").size().reset_index().rename(columns={0: "Size"}),
            pd.DataFrame.from_dict(latlon, orient="index")
            .reset_index()
            .rename(columns={"index": "PREFECTURE"}),
            on="PREFECTURE",
        )
        participate = participate.loc[:, ["PREFECTURE", "Size", "lat", "lon"]]
        st.dataframe(participate)


#############  LSI blood test processing ####################


def lsi_data_by_stress(data, lsi_value_item=None):

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Summary of {lsi_value_item}", f"{lsi_value_item} levels"),
        shared_xaxes=False,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}, {"type": "box"}]],
    )
    summary = summary_LSI_single_by_stress(data, lsi_value_item).round(2)

    table1 = go.Table(
        header=dict(
            values=[f for f in summary.columns],
            fill_color="paleturquoise",
            align="left",
            font_size=12,
        ),
        cells=dict(
            values=[
                summary.CELLNAME,
                summary["max"],
                summary["min"],
                summary["mean"],
                summary["median"],
                summary["std"],
            ],
            fill_color="lavender",
            align="left",
            font_size=12,
        ),
    )

    st.info(
        "See also: https://stackoverflow.com/questions/66890941/add-multiple-plotly-express-bar-figures-into-one-window"
    )
    trace1 = px.box(
        data.sort_values("CELLNAME"),
        x="CELLNAME",
        y=lsi_value_item,  #
        points="all",
        color="CELLNAME",
        color_discrete_sequence=["blue", "green", "red"],
        hover_data=["Age", "SAMPLEID", "Sex"]
        # name= lsi_value_item,
    )

    fig.add_traces(
        [table1, trace1["data"][0], trace1["data"][1], trace1["data"][2]],
        rows=[1, 1, 1, 1],
        cols=[1, 2, 2, 2],
    )
    fig.update_layout(width=1000, margin=dict(b=0.4, r=0.4, l=0))

    fig.update_traces(row=1, col=2, marker={"opacity": 0.4})
    # fig.update_
    return st.write(fig)


def lsi_by_multiple_variables(data, lsi_group_item_1st, lsi_value_item=None, **kwargs):
    """
    lsi_value_item = st.selectbox(
    "Select item",
    options = lsi.columns[3:].values.tolist()
    )
    """
    # st.write(data.sort_values(lsi_group_item_1st)[lsi_group_item_1st])  # Check a categroy order
    trace_1st = px.box(
        data.sort_values(["CELLNAME", lsi_group_item_1st]),
        x="CELLNAME",
        y=lsi_value_item,  #
        points="all",
        color=lsi_group_item_1st,
        # color_discrete_sequence = ["blue", "green","red"],
        hover_data=["Age", "SAMPLEID", "Sex"],
        category_orders={
            lsi_group_item_1st: data.sort_values(lsi_group_item_1st)[
                lsi_group_item_1st
            ].unique()
        }
        # name= lsi_value_item,
    )

    st.plotly_chart(trace_1st)


######


########################################################################


def main():
    st.title("Analysis App for the stress study in 2021")
    st.write("---")
    text = st.empty()
    text.warning("Please login before accessing the data")
    user_box = st.sidebar.empty()
    pwd_box = st.sidebar.empty()
    submit = st.sidebar.empty()
    user_box.text_input("Username", "", key="username")
    pwd_box.text_input("Password", "", key="password", type="password")
    st.session_state.checkbox_state = submit.checkbox("Login")
    if st.session_state.checkbox_state:
        # if password == '12345':
        create_usertable()
        hashed_pswd = make_hashes(st.session_state.password)
        result = login_user(
            st.session_state.username,
            check_hashes(st.session_state.password, hashed_pswd),
        )
        if result:
            user_box.empty()
            pwd_box.empty()
            submit.empty()
            text.success("Log in as {}".format(st.session_state.username))
            text.empty()
            logout_holder = st.sidebar.empty()
            logout = logout_holder.button("Log out")
            if logout:
                logout_holder.empty()
                text.success("Log out. Please refresh the page.")
                return
            # start your code here
            init_datapage()
        else:
            text.warning("Invalid Username or Password")


if __name__ == "__main__":
    main()
