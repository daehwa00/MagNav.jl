import h5py
import numpy as np
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt
from copy import deepcopy
from julia.api import Julia
from scipy.signal import medfilt

jl = Julia(compiled_modules=False)
from julia import Main

Main.eval(
    """
cd(@__DIR__)

using MagNav
using CSV, DataFrames
using Plots: plot, plot!
using Plots
using Random: seed!
using Statistics: mean, median, std
seed!(33);
include("dataframes_setup.jl");

flight = :Flt1006 # select flight, full list in df_flight
xyz    = get_XYZ(flight,df_flight); # load flight data

map_name   = :Eastern_395 # df_map중에서 지도 이름을 선택
df_options = df_nav[(df_nav.flight   .== flight  ) .&
                    (df_nav.map_name .== map_name),:]


line = 1006.08 # select flight line (row) from df_options
ind  = get_ind(xyz,line,df_options); # get Boolean indices


# lpf     = get_bpf(;pass1=0.0,pass2=0.2,fs=10.0) # get low-pass filter
# lpf_sig = -bpf_data(xyz.cur_strb[ind];bpf=lpf)  # apply low-pass filter, sign switched for easier comparison
# p5      = plot_basic(xyz.traj.tt[ind],lpf_sig;lab="filtered current for strobe lights"); # plot the low-pass filtered strobe light current sensor

"""
)


file_path = "/home/daehwa/.julia/artifacts/c3f3d3462514a6c52576cf8017a650ea9d10e64b/sgl_2020_train/Flt1002_train.h5"

# 데이터셋 이름 리스트
dataset_names = ["cur_com_1", "cur_strb", "cur_outpwr", "cur_ac_lo"]


def plot_data_with_lpf_and_impulse_removal(file_path, dataset_names, lpf):
    plt.figure(
        figsize=(30, 24)
    )  # 가로 크기를 늘려 새로운 그래프를 포함할 공간을 만듭니다.

    mean_differences = []

    for i, dataset_name in enumerate(dataset_names, 1):
        with h5py.File(file_path, "r") as file:
            data = file[dataset_name][:]

        # Python에서 Julia로 데이터 전달
        Main.data = data
        Main.lpf = lpf

        # Julia에서 데이터 처리
        Main.eval(
            """
            filtered_data = -bpf_data(data; bpf=lpf)
            """
        )

        # 결과를 Python으로 가져오기
        filtered_data = Main.filtered_data

        # 임펄스 노이즈 제거를 위한 중앙값 필터 적용
        # 여기서는 커널 사이즈를 예시로 3을 사용합니다.
        impulse_removed_data = medfilt(filtered_data, kernel_size=399)

        differences = impulse_removed_data - filtered_data
        # 차이의 평균을 계산하고 저장합니다.
        mean_difference = np.mean(differences)
        mean_differences.append(mean_difference)
        # 원본 데이터 플롯
        plt.subplot(len(dataset_names), 3, 3 * i - 2)
        plt.plot(data, label="Original " + dataset_name)
        plt.title("Original Data: " + dataset_name)
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()

        # 필터링된 데이터 플롯
        plt.subplot(len(dataset_names), 3, 3 * i - 1)
        plt.plot(filtered_data, label="Filtered " + dataset_name, color="orange")
        plt.title("Filtered Data: " + dataset_name)
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()

        # 임펄스 노이즈 제거된 데이터 플롯
        plt.subplot(len(dataset_names), 3, 3 * i)
        plt.plot(
            impulse_removed_data,
            label="Impulse Removed " + dataset_name,
            color="dodgerblue",
        )
        plt.title("Impulse Removed Data: " + dataset_name)
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.show()
    for dataset_name, mean_diff in zip(dataset_names, mean_differences):
        print(f"{dataset_name} mean difference: {mean_diff}")


# 데이터 플롯
# plot_data_with_lpf_and_impulse_removal(file_path, dataset_names, Main.lpf)

Main.eval(
    """
flts = [:Flt1003,:Flt1004,:Flt1005,:Flt1006] # 학습될 항공편
df_train = df_all[(df_all.flight .∈ (flts,) ) .& # df_all의 fligt 열에 있는 항목이 flts에 있는지 확인
                  (df_all.line   .!= line),:]    # 1006.08을 제외한 모든 항공편의 데이터를 선택
lines_train = df_train.line # training lines

TL_i   = 6 # select first calibration box of 1006.04
TL_ind = get_ind(xyz;tt_lim=[df_comp.t_start[TL_i],df_comp.t_end[TL_i]]);
λ       = 0.025   # ridge parameter for ridge regression
use_vec = :flux_d # selected vector (flux) magnetometer
flux    = getfield(xyz,use_vec) # load Flux D data
TL_d_4  = create_TL_coef(flux,xyz.mag_4_uc,TL_ind;λ=λ); # create Tolles-Lawson coefficients with Flux D & Mag 4

features = [:mag_4_uc, :lpf_cur_com_1, :lpf_cur_strb, :lpf_cur_outpwr, :lpf_cur_ac_lo];
comp_params = NNCompParams(features_setup = features,
                           model_type     = :m2c,
                           y_type         = :d,
                           use_mag        = :mag_4_uc,
                           use_vec        = :flux_d,
                           terms          = [:permanent,:induced,:fdm],
                           terms_A        = [:permanent,:induced,:eddy],
                           sub_diurnal    = true,
                           sub_igrf       = true,
                           bpf_mag        = false,
                           norm_type_A    = :none,
                           norm_type_x    = :standardize,
                           norm_type_y    = :standardize,
                           TL_coef        = TL_d_4,
                           η_adam         = 0.001,
                           epoch_adam     = 300,
                           epoch_lbfgs    = 0,
                           hidden         = [8,4]);
(comp_params,y_train,y_train_hat,err_train,feats) =
    comp_train(comp_params,lines_train,df_all,df_flight,df_map);

println(comp_params)
println(feats)
(_,y_hat,_) =
    comp_test(comp_params,[line],df_all,df_flight,df_map);

"""
)
