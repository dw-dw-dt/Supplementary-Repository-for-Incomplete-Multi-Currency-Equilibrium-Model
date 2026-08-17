import numpy as np
import pandas as pd

from utils import (
    calculate_alpha,
    get_multivariate_normal_sample,
    residual_systematic_resampler,
)

# 浮動小数点エラーの設定: divide, over を例外として扱う
np.seterr(divide="raise", over="raise", invalid="raise")


class ParticleFilter:
    def __init__(
        self,
        NUM_OF_COUNTRY,
        NUM_OF_RISKY_ASSET,
        DIM_OF_BM,
        NP,
        T,
        NUM_OF_dt_FOR_YEAR,
        dt,
        Q_D_0,
        BETA_D,
        BETA_F,
        Y1_0,
        MU_Y1,
        SIGMA_Y1,
        Y2_0,
        MU_Y2,
        SIGMA_Y2,
        Y3_0,
        MU_Y3,
        SIGMA_Y3,
        Y4_0,
        MU_Y4,
        SIGMA_Y4,
        MU_DELTA_D,
        MU_DELTA_F,
        SIGMA_DELTA_D,
        SIGMA_DELTA_F,
        X_D_0,
        X_F_0,
        GAMMA_DD,
        GAMMA_DF,
        GAMMA_FF,
        GAMMA_FD,
        STATE_NOISE_1,
        STATE_NOISE_2,
        STATE_NOISE_3,
        STATE_NOISE_4,
        OBS_NOISE_1,
        OBS_NOISE_2,
        OBS_NOISE_3,
        OBS_NOISE_4,
        OBS_NOISE_5,
        tp1_df,
        sp1_df,
        jprate_df,
        usrate_df,
        fx_df,
        d_1,
        d_2,
        d_3,
        d_4,
        f_1,
        f_2,
        f_3,
        f_4,
        ad_1,
        ad_2,
        ad_3,
        ad_4,
        bd_1,
        bd_2,
        bd_3,
        bd_4,
        af_1,
        af_2,
        af_3,
        af_4,
        bf_1,
        bf_2,
        bf_3,
        bf_4,
        EPSILON=1e-6,
        T_hat=5.0,
    ):
        """
        Particle Filterの初期化
        d_1, d_2, d_3, d_4, f_1, f_2, f_3, f_4は、各国のμ^δのドリフト用
        """
        self.NUM_OF_COUNTRY = NUM_OF_COUNTRY
        self.NUM_OF_RISKY_ASSET = NUM_OF_RISKY_ASSET
        self.DIM_OF_BM = DIM_OF_BM
        self.NP = NP
        self.T = T
        self.NUM_OF_dt_FOR_YEAR = NUM_OF_dt_FOR_YEAR
        self.dt = dt
        self.Q_D_0 = Q_D_0
        self.BETA_D = BETA_D
        self.BETA_F = BETA_F

        self.Y1_0 = Y1_0
        self.MU_Y1 = MU_Y1
        self.SIGMA_Y1 = SIGMA_Y1
        self.Y2_0 = Y2_0
        self.MU_Y2 = MU_Y2
        self.SIGMA_Y2 = SIGMA_Y2
        self.Y3_0 = Y3_0
        self.MU_Y3 = MU_Y3
        self.SIGMA_Y3 = SIGMA_Y3
        self.Y4_0 = Y4_0
        self.MU_Y4 = MU_Y4
        self.SIGMA_Y4 = SIGMA_Y4

        self.MU_DELTA_D = MU_DELTA_D
        self.MU_DELTA_F = MU_DELTA_F
        self.SIGMA_DELTA_D = SIGMA_DELTA_D
        self.SIGMA_DELTA_F = SIGMA_DELTA_F

        self.X_D_0 = X_D_0
        self.X_F_0 = X_F_0
        self.GAMMA_DD = GAMMA_DD
        self.GAMMA_DF = GAMMA_DF
        self.GAMMA_FF = GAMMA_FF
        self.GAMMA_FD = GAMMA_FD

        self.STATE_NOISE_1 = STATE_NOISE_1
        self.STATE_NOISE_2 = STATE_NOISE_2
        self.STATE_NOISE_3 = STATE_NOISE_3
        self.STATE_NOISE_4 = STATE_NOISE_4
        self.OBS_NOISE_1 = OBS_NOISE_1
        self.OBS_NOISE_2 = OBS_NOISE_2
        self.OBS_NOISE_3 = OBS_NOISE_3
        self.OBS_NOISE_4 = OBS_NOISE_4
        self.OBS_NOISE_5 = OBS_NOISE_5

        self.d_1 = d_1
        self.d_2 = d_2
        self.d_3 = d_3
        self.d_4 = d_4
        self.f_1 = f_1
        self.f_2 = f_2
        self.f_3 = f_3
        self.f_4 = f_4

        self.ad_1 = ad_1
        self.ad_2 = ad_2
        self.ad_3 = ad_3
        self.ad_4 = ad_4
        self.bd_1 = bd_1
        self.bd_2 = bd_2
        self.bd_3 = bd_3
        self.bd_4 = bd_4
        self.af_1 = af_1
        self.af_2 = af_2
        self.af_3 = af_3
        self.af_4 = af_4
        self.bf_1 = bf_1
        self.bf_2 = bf_2
        self.bf_3 = bf_3
        self.bf_4 = bf_4

        # 上記に依存して決まる初期値; deltaの初期値はconsumption market clearing conditionを満たすように設定
        self.Q_F_0 = 1.0 / Q_D_0
        self.DELTA_D_0 = GAMMA_DD * BETA_D * X_D_0 / (
            1 - np.exp(-BETA_D * T)
        ) + GAMMA_FD * BETA_F * X_F_0 * Q_D_0 / (1 - np.exp(-BETA_F * T))
        self.DELTA_F_0 = GAMMA_DF * BETA_D * X_D_0 / (
            1 - np.exp(-BETA_D * T)
        ) / Q_D_0 + GAMMA_FF * BETA_F * X_F_0 / (1 - np.exp(-BETA_F * T))
        self.Y_D = (1 - np.exp(-BETA_D * T)) / (BETA_D * X_D_0)
        self.Y_F = (1 - np.exp(-BETA_F * T)) / (BETA_F * X_F_0)

        self.tp1_df = tp1_df
        self.sp1_df = sp1_df
        self.jprate_df = jprate_df
        self.usrate_df = usrate_df
        self.fx_df = fx_df

        self.EPSILON = EPSILON
        self.T_hat = T_hat

    def calc_drift_of_delta_d(self, y_1, y_2, y_3, y_4):
        return self.MU_DELTA_D * (
            self.d_1 * y_1 + self.d_2 * y_2 + self.d_3 * y_3 + self.d_4 * y_4
        )

    def calc_drift_of_delta_f(self, y_1, y_2, y_3, y_4):
        return self.MU_DELTA_F * (
            self.f_1 * y_1 + self.f_2 * y_2 + self.f_3 * y_3 + self.f_4 * y_4
        )

    def calc_S_D(
        self,
        delta_d,
        x_d_0,
        gamma_dd,
        gamma_fd,
        beta_d,
        q_d_0,
        x_f_0,
        beta_f,
        z_d,
        z_f,
        t,
        Tau,
    ):
        # 日本の株価
        Z_D = z_d
        Z_F = z_f
        K_DD = (
            x_d_0
            * gamma_dd
            * beta_d
            * np.exp(-beta_d * t)
            / (1 - np.exp(-beta_d * self.T))
        )
        K_DF = (
            q_d_0
            * x_f_0
            * gamma_fd
            * beta_f
            * np.exp(-beta_f * t)
            / (1 - np.exp(-beta_f * self.T))
        )
        Z_DF = (
            (1 - np.exp(-beta_d * Tau)) / beta_d * K_DD * Z_D
            + (1 - np.exp(-beta_f * Tau)) / beta_f * K_DF * Z_F
        ) / (K_DD * Z_D + K_DF * Z_F)
        S_D = delta_d * Z_DF
        return S_D

    def calc_S_F(
        self,
        delta_f,
        x_d_0,
        gamma_df,
        gamma_ff,
        beta_d,
        q_d_0,
        x_f_0,
        beta_f,
        z_d,
        z_f,
        t,
        Tau,
    ):
        # 米国の株価
        Z_D = z_d
        Z_F = z_f
        K_FD = (
            x_d_0
            / q_d_0
            * gamma_df
            * beta_d
            * np.exp(-beta_d * t)
            / (1 - np.exp(-beta_d * self.T))
        )
        K_FF = (
            x_f_0
            * gamma_ff
            * beta_f
            * np.exp(-beta_f * t)
            / (1 - np.exp(-beta_f * self.T))
        )
        Z_FD = (
            (1 - np.exp(-beta_d * Tau)) / beta_d * K_FD * Z_D
            + (1 - np.exp(-beta_f * Tau)) / beta_f * K_FF * Z_F
        ) / (K_FD * Z_D + K_FF * Z_F)
        S_F = delta_f * Z_FD
        return S_F

    def calc_R_D(
        self,
        drift_of_delta_d,
        sigma_delta_d,
        alpha_d,
        gamma_dd,
        q_d_0,
        y_d,
        alpha_f,
        gamma_fd,
        y_f,
        beta_d,
        beta_f,
        lambda_hat_d,
        lambda_hat_f,
        z_d,
        z_f,
    ):
        # 日本の金利
        Z_D = z_d
        Z_F = z_f
        denominator = (alpha_d * gamma_dd / y_d) * Z_D + (
            alpha_f * gamma_fd * q_d_0 / y_f
        ) * Z_F
        weighted_beta = (
            (alpha_d * gamma_dd / y_d) * Z_D * beta_d
            + (alpha_f * gamma_fd * q_d_0 / y_f) * Z_F * beta_f
        ) / denominator
        weighted_lambda_hat = (
            (alpha_d * gamma_dd / y_d) * Z_D * lambda_hat_d
            + (alpha_f * gamma_fd * q_d_0 / y_f) * Z_F * lambda_hat_f
        ) / denominator
        R_D = (
            drift_of_delta_d
            - (sigma_delta_d.T @ sigma_delta_d).item()
            + weighted_beta
            + (weighted_lambda_hat @ sigma_delta_d)
        )
        return R_D

    def calc_R_F(
        self,
        drift_of_delta_f,
        sigma_delta_f,
        alpha_d,
        gamma_df,
        q_f_0,
        y_d,
        alpha_f,
        gamma_ff,
        y_f,
        beta_d,
        beta_f,
        lambda_hat_d,
        lambda_hat_f,
        z_d,
        z_f,
    ):
        Z_D = z_d
        Z_F = z_f
        denominator = (alpha_d * gamma_df * q_f_0 / y_d) * Z_D + (
            alpha_f * gamma_ff / y_f
        ) * Z_F
        weighted_beta = (
            (alpha_d * gamma_df * q_f_0 / y_d) * Z_D * beta_d
            + (alpha_f * gamma_ff / y_f) * Z_F * beta_f
        ) / denominator
        weighted_lambda_hat = (
            (alpha_d * gamma_df * q_f_0 / y_d) * Z_D * lambda_hat_d
            + (alpha_f * gamma_ff / y_f) * Z_F * lambda_hat_f
        ) / denominator
        R_F = (
            drift_of_delta_f
            - (sigma_delta_f.T @ sigma_delta_f).item()
            + weighted_beta
            + (weighted_lambda_hat @ sigma_delta_f)
        )
        return R_F

    def calc_THETA_D(
        self,
        sigma_delta_d,
        alpha_d,
        gamma_dd,
        q_d_0,
        y_d,
        alpha_f,
        gamma_fd,
        y_f,
        lambda_hat_d,
        lambda_hat_f,
        z_d,
        z_f,
    ):
        # 日本のmarket price of risk
        Z_D = z_d
        Z_F = z_f
        denominator = (alpha_d * gamma_dd / y_d) * Z_D + (
            alpha_f * gamma_fd * q_d_0 / y_f
        ) * Z_F
        weighted_lambda_hat = (
            (alpha_d * gamma_dd / y_d) * Z_D * lambda_hat_d
            + (alpha_f * gamma_fd * q_d_0 / y_f) * Z_F * lambda_hat_f
        ) / denominator
        THETA_D = sigma_delta_d.T - weighted_lambda_hat
        return THETA_D

    def calc_THETA_F(
        self,
        sigma_delta_f,
        alpha_d,
        gamma_df,
        q_f_0,
        y_d,
        alpha_f,
        gamma_ff,
        y_f,
        lambda_hat_d,
        lambda_hat_f,
        z_d,
        z_f,
    ):
        Z_D = z_d
        Z_F = z_f
        denominator = (alpha_d * gamma_df * q_f_0 / y_d) * Z_D + (
            alpha_f * gamma_ff / y_f
        ) * Z_F
        weighted_lambda_hat = (
            (alpha_d * gamma_df * q_f_0 / y_d) * Z_D * lambda_hat_d
            + (alpha_f * gamma_ff / y_f) * Z_F * lambda_hat_f
        ) / denominator
        THETA_F = sigma_delta_f.T - weighted_lambda_hat
        return THETA_F

    def filtering(self):
        # 状態空間モデルの推定用
        # 1期先予測配列 を作成
        Y_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が Y の次元
        DELTA_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 2], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が delta の次元
        Z_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 2], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が z の次元
        Q_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 1], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が q の次元
        S_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 2], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が s の次元
        R_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 2], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が r の次元
        THETA_D_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が theta の次元
        THETA_F_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が theta の次元
        LAMBDA_HAT_D_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が theta の次元
        LAMBDA_HAT_F_pred = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が theta の次元

        # 状態変数の filtered配列 を作成
        Y_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が Y の次元
        DELTA_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 2], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が delta の次元
        Z_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 2], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が z の次元
        Q_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 1], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が q の次元
        S_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 2], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が s の次元
        R_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, 2], dtype=np.float32
        )  # 1次元目が時間、2次元目がparticle、3次元目が r の次元
        THETA_D_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が theta の次元
        THETA_F_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が theta の次元
        LAMBDA_HAT_D_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が theta の次元
        LAMBDA_HAT_F_filt = np.zeros(
            [self.T * self.NUM_OF_dt_FOR_YEAR, self.NP, self.DIM_OF_BM],
            dtype=np.float32,
        )  # 1次元目が時間、2次元目がparticle、3次元目が theta の次元

        # t=0の初期値を pred データフレームに投入
        Y_pred[0, :, :] = np.array([self.Y1_0, self.Y2_0, self.Y3_0, self.Y4_0])
        DELTA_pred[0, :, :] = np.array([self.DELTA_D_0, self.DELTA_F_0])
        Z_pred[0, :, :] = np.array([1.0, 1.0])
        Q_pred[0, :, :] = np.array([self.Q_D_0])
        # t=0の初期値を filt データフレームに投入
        Y_filt[0, :, :] = Y_pred[0, :, :].copy()
        DELTA_filt[0, :, :] = DELTA_pred[0, :, :].copy()
        Z_filt[0, :, :] = Z_pred[0, :, :].copy()
        Q_filt[0, :, :] = Q_pred[0, :, :].copy()

        # Log likelihood
        llh = np.zeros([self.T * self.NUM_OF_dt_FOR_YEAR, self.NP], dtype=np.float32)

        # Resampling
        res = np.zeros(self.NP, dtype=int)

        param_df = pd.DataFrame(
            {
                # 時間
                "t": [0],
                # Y
                "Y1_t": [self.Y1_0],
                "Y2_t": [self.Y2_0],
                "Y3_t": [self.Y3_0],
                "Y4_t": [self.Y4_0],
                # \delta のドリフトと値
                "DRIFT_OF_DELTA_D_t": [
                    self.calc_drift_of_delta_d(
                        y_1=self.Y1_0, y_2=self.Y2_0, y_3=self.Y3_0, y_4=self.Y4_0
                    )
                ],
                "DRIFT_OF_DELTA_F_t": [
                    self.calc_drift_of_delta_f(
                        y_1=self.Y1_0, y_2=self.Y2_0, y_3=self.Y3_0, y_4=self.Y4_0
                    )
                ],
                "DELTA_D_t": [self.DELTA_D_0],
                "DELTA_F_t": [self.DELTA_F_0],
                # 初期資産
                "X_D_t": [self.X_D_0],
                "X_F_t": [self.X_F_0],
                # 以下内生的に決まる
                # 株価（内生的に決まる）
                "S_D_t": [np.nan],
                "S_F_t": [np.nan],
                # interest rate
                "R_D_t": [np.nan],
                "R_F_t": [np.nan],
                # market price of risk（内生的に決まる）
                "THETA_D_t": [np.zeros(self.DIM_OF_BM, dtype=np.float32)],
                "THETA_F_t": [np.zeros(self.DIM_OF_BM, dtype=np.float32)],
                # \lambda_hat
                "LAMBDA_HAT_D_t": [np.zeros(self.DIM_OF_BM, dtype=np.float32)],
                "LAMBDA_HAT_F_t": [np.zeros(self.DIM_OF_BM, dtype=np.float32)],
                # Z
                "Z_D_t": [1],
                "Z_F_t": [1],
                # 為替レート（内生的に決まる）
                "Q_D_t": [self.Q_D_0],
                "Q_F_t": [self.Q_F_0],
                # optimal portfolio（内生的に決まる）
                "PI_D_t": [np.zeros(self.DIM_OF_BM, dtype=np.float32)],
                "PI_F_t": [np.zeros(self.DIM_OF_BM, dtype=np.float32)],
            }
        )

        for i in range(0, self.T * self.NUM_OF_dt_FOR_YEAR + 1):
            t = i * self.dt
            if t >= self.T - self.T_hat:
                break  # 22年ごろまでしかデータが無いので
            Tau = self.T - t
            param_df.loc[i, "t"] = t
            alpha_D = calculate_alpha(beta=self.BETA_D, t=t)
            alpha_F = calculate_alpha(beta=self.BETA_F, t=t)

            if i == 0:
                # 状態変数のフィルタリング値を取得, 計算
                Y1_t = Y_filt[i, :, 0].reshape(self.NP, 1)
                Y2_t = Y_filt[i, :, 1].reshape(self.NP, 1)
                Y3_t = Y_filt[i, :, 2].reshape(self.NP, 1)
                Y4_t = Y_filt[i, :, 3].reshape(self.NP, 1)
                DELTA_D_t = DELTA_filt[i, :, 0].reshape(self.NP, 1)
                DELTA_F_t = DELTA_filt[i, :, 1].reshape(self.NP, 1)
                Z_D_t = Z_filt[i, :, 0].reshape(self.NP, 1)
                Z_F_t = Z_filt[i, :, 1].reshape(self.NP, 1)
                Q_D_t = Q_filt[i, :, :].reshape(self.NP, 1)
                LAMBDA_HAT_D_t = LAMBDA_HAT_D_filt[i, :, :].reshape(
                    self.NP, self.DIM_OF_BM
                )
                LAMBDA_HAT_F_t = LAMBDA_HAT_F_filt[i, :, :].reshape(
                    self.NP, self.DIM_OF_BM
                )

                drift_of_delta_d = self.calc_drift_of_delta_d(
                    y_1=Y1_t, y_2=Y2_t, y_3=Y3_t, y_4=Y4_t
                )
                drift_of_delta_f = self.calc_drift_of_delta_f(
                    y_1=Y1_t, y_2=Y2_t, y_3=Y3_t, y_4=Y4_t
                )

                # 観測変数を値を計算する（Qは初期値があるので計算不要）
                S_D_t = self.calc_S_D(
                    delta_d=DELTA_D_t,
                    x_d_0=self.X_D_0,
                    gamma_dd=self.GAMMA_DD,
                    gamma_fd=self.GAMMA_FD,
                    beta_d=self.BETA_D,
                    q_d_0=self.Q_D_0,
                    x_f_0=self.X_F_0,
                    beta_f=self.BETA_F,
                    z_d=Z_D_t,
                    z_f=Z_F_t,
                    t=t,
                    Tau=Tau,
                ).reshape(self.NP)
                S_F_t = self.calc_S_F(
                    delta_f=DELTA_F_t,
                    x_d_0=self.X_D_0,
                    gamma_df=self.GAMMA_DF,
                    gamma_ff=self.GAMMA_FF,
                    beta_d=self.BETA_D,
                    q_d_0=self.Q_D_0,
                    x_f_0=self.X_F_0,
                    beta_f=self.BETA_F,
                    z_d=Z_D_t,
                    z_f=Z_F_t,
                    t=t,
                    Tau=Tau,
                ).reshape(self.NP)
                S_filt[0, :, 0] = S_D_t
                S_filt[0, :, 1] = S_F_t
                R_D_t = self.calc_R_D(
                    drift_of_delta_d=drift_of_delta_d,
                    sigma_delta_d=self.SIGMA_DELTA_D,
                    alpha_d=alpha_D,
                    gamma_dd=self.GAMMA_DD,
                    q_d_0=self.Q_D_0,
                    y_d=self.Y_D,
                    alpha_f=alpha_F,
                    gamma_fd=self.GAMMA_FD,
                    y_f=self.Y_F,
                    beta_d=self.BETA_D,
                    beta_f=self.BETA_F,
                    lambda_hat_d=LAMBDA_HAT_D_t,
                    lambda_hat_f=LAMBDA_HAT_F_t,
                    z_d=Z_D_t,
                    z_f=Z_F_t,
                ).reshape(self.NP)
                R_F_t = self.calc_R_F(
                    drift_of_delta_f=drift_of_delta_f,
                    sigma_delta_f=self.SIGMA_DELTA_F,
                    alpha_d=alpha_D,
                    gamma_df=self.GAMMA_DF,
                    q_f_0=self.Q_F_0,
                    y_d=self.Y_D,
                    alpha_f=alpha_F,
                    gamma_ff=self.GAMMA_FF,
                    y_f=self.Y_F,
                    beta_d=self.BETA_D,
                    beta_f=self.BETA_F,
                    lambda_hat_d=LAMBDA_HAT_D_t,
                    lambda_hat_f=LAMBDA_HAT_F_t,
                    z_d=Z_D_t,
                    z_f=Z_F_t,
                ).reshape(self.NP)
                R_filt[0, :, 0] = R_D_t
                R_filt[0, :, 1] = R_F_t

                # その他の状態変数の計算
                THETA_D_t = self.calc_THETA_D(
                    sigma_delta_d=self.SIGMA_DELTA_D,
                    alpha_d=alpha_D,
                    gamma_dd=self.GAMMA_DD,
                    q_d_0=self.Q_D_0,
                    y_d=self.Y_D,
                    alpha_f=alpha_F,
                    gamma_fd=self.GAMMA_FD,
                    y_f=self.Y_F,
                    lambda_hat_d=LAMBDA_HAT_D_t,
                    lambda_hat_f=LAMBDA_HAT_F_t,
                    z_d=Z_D_t,
                    z_f=Z_F_t,
                ).reshape(self.NP, self.DIM_OF_BM)
                THETA_F_t = self.calc_THETA_F(
                    sigma_delta_f=self.SIGMA_DELTA_F,
                    alpha_d=alpha_D,
                    gamma_df=self.GAMMA_DF,
                    q_f_0=self.Q_F_0,
                    y_d=self.Y_D,
                    alpha_f=alpha_F,
                    gamma_ff=self.GAMMA_FF,
                    y_f=self.Y_F,
                    lambda_hat_d=LAMBDA_HAT_D_t,
                    lambda_hat_f=LAMBDA_HAT_F_t,
                    z_d=Z_D_t,
                    z_f=Z_F_t,
                ).reshape(self.NP, self.DIM_OF_BM)
                THETA_D_filt[0, :, :] = THETA_D_t
                THETA_F_filt[0, :, :] = THETA_F_t

                # テーブル更新
                param_df.at[0, "S_D_t"] = np.mean(S_D_t)
                param_df.at[0, "S_F_t"] = np.mean(S_F_t)
                param_df.at[0, "R_D_t"] = np.mean(R_D_t)
                param_df.at[0, "R_F_t"] = np.mean(R_F_t)
                param_df.at[0, "THETA_D_t"] = np.mean(
                    THETA_D_t, axis=0
                )  # locだと複数セルへの代入になるので、atを使用。atは単一セルへの代入。
                param_df.at[0, "THETA_F_t"] = np.mean(THETA_F_t, axis=0)

                # 正規化用
                S_D_0 = np.mean(S_D_t)
                S_F_0 = np.mean(S_F_t)
                R_D_0 = np.mean(R_D_t)
                R_F_0 = np.mean(R_F_t)
            else:
                # pred値の取得
                Y1_t_pred = Y_pred[i, :, 0].reshape(self.NP, 1)
                Y2_t_pred = Y_pred[i, :, 1].reshape(self.NP, 1)
                Y3_t_pred = Y_pred[i, :, 2].reshape(self.NP, 1)
                Y4_t_pred = Y_pred[i, :, 3].reshape(self.NP, 1)
                DELTA_D_t_pred = DELTA_pred[i, :, 0].reshape(self.NP, 1)
                DELTA_F_t_pred = DELTA_pred[i, :, 1].reshape(self.NP, 1)
                Z_D_t_pred = Z_pred[i, :, 0].reshape(self.NP, 1)
                Z_F_t_pred = Z_pred[i, :, 1].reshape(self.NP, 1)
                LAMBDA_HAT_D_t_pred = LAMBDA_HAT_D_pred[i, :, :].reshape(
                    self.NP, self.DIM_OF_BM
                )
                LAMBDA_HAT_F_t_pred = LAMBDA_HAT_F_pred[i, :, :].reshape(
                    self.NP, self.DIM_OF_BM
                )

                drift_of_delta_d_pred = self.calc_drift_of_delta_d(
                    y_1=Y1_t_pred, y_2=Y2_t_pred, y_3=Y3_t_pred, y_4=Y4_t_pred
                )
                drift_of_delta_f_pred = self.calc_drift_of_delta_f(
                    y_1=Y1_t_pred, y_2=Y2_t_pred, y_3=Y3_t_pred, y_4=Y4_t_pred
                )

                # 観測変数の計算
                S_D_t_pred = self.calc_S_D(
                    delta_d=DELTA_D_t_pred,
                    x_d_0=self.X_D_0,
                    gamma_dd=self.GAMMA_DD,
                    gamma_fd=self.GAMMA_FD,
                    beta_d=self.BETA_D,
                    q_d_0=self.Q_D_0,
                    x_f_0=self.X_F_0,
                    beta_f=self.BETA_F,
                    z_d=Z_D_t_pred,
                    z_f=Z_F_t_pred,
                    t=t,
                    Tau=Tau,
                ).reshape(self.NP)
                S_F_t_pred = self.calc_S_F(
                    delta_f=DELTA_F_t_pred,
                    x_d_0=self.X_D_0,
                    gamma_df=self.GAMMA_DF,
                    gamma_ff=self.GAMMA_FF,
                    beta_d=self.BETA_D,
                    q_d_0=self.Q_D_0,
                    x_f_0=self.X_F_0,
                    beta_f=self.BETA_F,
                    z_d=Z_D_t_pred,
                    z_f=Z_F_t_pred,
                    t=t,
                    Tau=Tau,
                ).reshape(self.NP)
                S_pred[i, :, 0] = S_D_t_pred
                S_pred[i, :, 1] = S_F_t_pred
                R_D_t_pred = self.calc_R_D(
                    drift_of_delta_d=drift_of_delta_d_pred,
                    sigma_delta_d=self.SIGMA_DELTA_D,
                    alpha_d=alpha_D,
                    gamma_dd=self.GAMMA_DD,
                    q_d_0=self.Q_D_0,
                    y_d=self.Y_D,
                    alpha_f=alpha_F,
                    gamma_fd=self.GAMMA_FD,
                    y_f=self.Y_F,
                    beta_d=self.BETA_D,
                    beta_f=self.BETA_F,
                    lambda_hat_d=LAMBDA_HAT_D_t_pred,
                    lambda_hat_f=LAMBDA_HAT_F_t_pred,
                    z_d=Z_D_t_pred,
                    z_f=Z_F_t_pred,
                ).reshape(self.NP)
                R_F_t_pred = self.calc_R_F(
                    drift_of_delta_f=drift_of_delta_f_pred,
                    sigma_delta_f=self.SIGMA_DELTA_F,
                    alpha_d=alpha_D,
                    gamma_df=self.GAMMA_DF,
                    q_f_0=self.Q_F_0,
                    y_d=self.Y_D,
                    alpha_f=alpha_F,
                    gamma_ff=self.GAMMA_FF,
                    y_f=self.Y_F,
                    beta_d=self.BETA_D,
                    beta_f=self.BETA_F,
                    lambda_hat_d=LAMBDA_HAT_D_t_pred,
                    lambda_hat_f=LAMBDA_HAT_F_t_pred,
                    z_d=Z_D_t_pred,
                    z_f=Z_F_t_pred,
                ).reshape(self.NP)
                R_pred[i, :, 0] = R_D_t_pred
                R_pred[i, :, 1] = R_F_t_pred
                Q_t_pred = Q_pred[i, :, 0]
                ## THETAも併せて計算
                THETA_D_t_pred = self.calc_THETA_D(
                    sigma_delta_d=self.SIGMA_DELTA_D,
                    alpha_d=alpha_D,
                    gamma_dd=self.GAMMA_DD,
                    q_d_0=self.Q_D_0,
                    y_d=self.Y_D,
                    alpha_f=alpha_F,
                    gamma_fd=self.GAMMA_FD,
                    y_f=self.Y_F,
                    lambda_hat_d=LAMBDA_HAT_D_t_pred,
                    lambda_hat_f=LAMBDA_HAT_F_t_pred,
                    z_d=Z_D_t_pred,
                    z_f=Z_F_t_pred,
                ).reshape(self.NP, self.DIM_OF_BM)
                THETA_F_t_pred = self.calc_THETA_F(
                    sigma_delta_f=self.SIGMA_DELTA_F,
                    alpha_d=alpha_D,
                    gamma_df=self.GAMMA_DF,
                    q_f_0=self.Q_F_0,
                    y_d=self.Y_D,
                    alpha_f=alpha_F,
                    gamma_ff=self.GAMMA_FF,
                    y_f=self.Y_F,
                    lambda_hat_d=LAMBDA_HAT_D_t_pred,
                    lambda_hat_f=LAMBDA_HAT_F_t_pred,
                    z_d=Z_D_t_pred,
                    z_f=Z_F_t_pred,
                ).reshape(self.NP, self.DIM_OF_BM)
                THETA_D_pred[i, :, :] = THETA_D_t_pred
                THETA_F_pred[i, :, :] = THETA_F_t_pred

                # 尤度計算
                # s1, s2, jp_rate, us_rate, fx最適化
                obs_s1 = self.tp1_df.loc[i, "scaled_value"]
                obs_s2 = self.sp1_df.loc[i, "scaled_value"]
                obs_r1 = self.jprate_df.loc[i, "scaled_value"]
                obs_r2 = self.usrate_df.loc[i, "scaled_value"]
                obs_fx = self.fx_df.loc[i, "scaled_value"]
                llh[i] = (
                    -2 * np.log(2 * np.pi)
                    - np.log(self.OBS_NOISE_1)
                    - np.log(self.OBS_NOISE_2)
                    - np.log(self.OBS_NOISE_3)
                    - np.log(self.OBS_NOISE_4)
                    - np.log(self.OBS_NOISE_5)
                    - 0.5
                    * (
                        (obs_s1 - S_D_t_pred / S_D_0) ** 2 / (self.OBS_NOISE_1**2)
                        + (obs_s2 - S_F_t_pred / S_F_0) ** 2 / (self.OBS_NOISE_2**2)
                        + (obs_r1 - R_D_t_pred / R_D_0) ** 2 / (self.OBS_NOISE_3**2)
                        + (obs_r2 - R_F_t_pred / R_F_0) ** 2 / (self.OBS_NOISE_4**2)
                        + (obs_fx - Q_t_pred) ** 2 / (self.OBS_NOISE_5**2)
                    )
                ).reshape(self.NP)

                # フィルタリング処理
                ## weight of each particle
                weight = np.exp(llh[i] - np.max(llh[i])) / np.sum(
                    np.exp(llh[i] - np.max(llh[i]))
                )

                ## systematic residual resampling
                res = residual_systematic_resampler(
                    Np=self.NP, weight=weight, RANDOM_SEED=i
                )

                ## filtering
                l = 0
                for j in range(self.NP):
                    for k in range(res[j]):
                        Y_filt[i, l] = Y_pred[i, j]
                        DELTA_filt[i, l] = DELTA_pred[i, j]
                        Z_filt[i, l] = Z_pred[i, j]
                        Q_filt[i, l] = Q_pred[i, j]
                        S_filt[i, l] = np.array([S_D_t_pred[j], S_F_t_pred[j]])
                        R_filt[i, l] = np.array([R_D_t_pred[j], R_F_t_pred[j]])
                        THETA_D_filt[i, l] = THETA_D_pred[i, j]
                        THETA_F_filt[i, l] = THETA_F_pred[i, j]
                        LAMBDA_HAT_D_filt[i, l] = LAMBDA_HAT_D_pred[i, j]
                        LAMBDA_HAT_F_filt[i, l] = LAMBDA_HAT_F_pred[i, j]
                        l += 1

                # データをテーブルに格納
                Y1_t = Y_filt[i, :, 0].reshape(self.NP, 1)
                Y2_t = Y_filt[i, :, 1].reshape(self.NP, 1)
                Y3_t = Y_filt[i, :, 2].reshape(self.NP, 1)
                Y4_t = Y_filt[i, :, 3].reshape(self.NP, 1)
                param_df.at[i, "Y1_t"] = np.mean(Y1_t)
                param_df.at[i, "Y2_t"] = np.mean(Y2_t)
                param_df.at[i, "Y3_t"] = np.mean(Y3_t)
                param_df.at[i, "Y4_t"] = np.mean(Y4_t)

                drift_of_delta_d = self.calc_drift_of_delta_d(
                    y_1=Y1_t, y_2=Y2_t, y_3=Y3_t, y_4=Y4_t
                )
                drift_of_delta_f = self.calc_drift_of_delta_f(
                    y_1=Y1_t, y_2=Y2_t, y_3=Y3_t, y_4=Y4_t
                )
                param_df.at[i, "DRIFT_OF_DELTA_D_t"] = np.mean(drift_of_delta_d)
                param_df.at[i, "DRIFT_OF_DELTA_F_t"] = np.mean(drift_of_delta_f)

                DELTA_D_t = DELTA_filt[i, :, 0].reshape(self.NP, 1)
                DELTA_F_t = DELTA_filt[i, :, 1].reshape(self.NP, 1)
                param_df.at[i, "DELTA_D_t"] = np.mean(DELTA_D_t)
                param_df.at[i, "DELTA_F_t"] = np.mean(DELTA_F_t)

                S_D_t = S_filt[i, :, 0].reshape(self.NP, 1)
                S_F_t = S_filt[i, :, 1].reshape(self.NP, 1)
                param_df.at[i, "S_D_t"] = np.mean(S_D_t)
                param_df.at[i, "S_F_t"] = np.mean(S_F_t)

                R_D_t = R_filt[i, :, 0].reshape(self.NP, 1)
                R_F_t = R_filt[i, :, 1].reshape(self.NP, 1)
                param_df.at[i, "R_D_t"] = np.mean(R_D_t)
                param_df.at[i, "R_F_t"] = np.mean(R_F_t)

                THETA_D_t = THETA_D_filt[i, :, :].reshape(self.NP, self.DIM_OF_BM)
                THETA_F_t = THETA_F_filt[i, :, :].reshape(self.NP, self.DIM_OF_BM)
                param_df.at[i, "THETA_D_t"] = np.mean(THETA_D_t, axis=0)
                param_df.at[i, "THETA_F_t"] = np.mean(THETA_F_t, axis=0)

                LAMBDA_HAT_D_t = LAMBDA_HAT_D_filt[i, :, :].reshape(
                    self.NP, self.DIM_OF_BM
                )
                LAMBDA_HAT_F_t = LAMBDA_HAT_F_filt[i, :, :].reshape(
                    self.NP, self.DIM_OF_BM
                )
                param_df.at[i, "LAMBDA_HAT_D_t"] = np.mean(LAMBDA_HAT_D_t, axis=0)
                param_df.at[i, "LAMBDA_HAT_F_t"] = np.mean(LAMBDA_HAT_F_t, axis=0)

                Q_D_t = Q_filt[i, :, :].reshape(self.NP, 1)
                param_df.at[i, "Q_D_t"] = np.mean(Q_D_t)
                # param_df.at[i, "Q_F_t"] = 1 / np.mean(Q_D_t)

                Z_D_t = Z_filt[i, :, 0].reshape(self.NP, 1)
                Z_F_t = Z_filt[i, :, 1].reshape(self.NP, 1)
                param_df.at[i, "Z_D_t"] = np.mean(Z_D_t)
                param_df.at[i, "Z_F_t"] = np.mean(Z_F_t)

            # t -> t+dt のpredの計算
            dW = get_multivariate_normal_sample(
                RANDOM_SEED=i,
                dim=self.DIM_OF_BM,
                dist_mean=np.array([0.0] * self.DIM_OF_BM),
                dist_cov=np.diag(
                    [
                        self.dt * self.STATE_NOISE_1,
                        self.dt * self.STATE_NOISE_2,
                        self.dt * self.STATE_NOISE_3,
                        self.dt * self.STATE_NOISE_4,
                    ]
                ),
                size=self.NP,
            )

            Y_pred[i + 1, :, 0] = (
                Y1_t - self.MU_Y1 * Y1_t * self.dt + dW @ self.SIGMA_Y1
            ).reshape(self.NP)
            Y_pred[i + 1, :, 1] = (
                Y2_t - self.MU_Y2 * Y2_t * self.dt + dW @ self.SIGMA_Y2
            ).reshape(self.NP)
            Y_pred[i + 1, :, 2] = (
                Y3_t - self.MU_Y3 * Y3_t * self.dt + dW @ self.SIGMA_Y3
            ).reshape(self.NP)
            Y_pred[i + 1, :, 3] = (
                Y4_t - self.MU_Y4 * Y4_t * self.dt + dW @ self.SIGMA_Y4
            ).reshape(self.NP)
            DELTA_pred[i + 1, :, 0] = (
                DELTA_D_t
                + DELTA_D_t * (drift_of_delta_d * self.dt + dW @ self.SIGMA_DELTA_D)
            ).reshape(self.NP)
            DELTA_pred[i + 1, :, 1] = (
                DELTA_F_t
                + DELTA_F_t * (drift_of_delta_f * self.dt + dW @ self.SIGMA_DELTA_F)
            ).reshape(self.NP)
            Z_pred[i + 1, :, 0] = (
                Z_D_t + Z_D_t * (dW * LAMBDA_HAT_D_t).sum(axis=1, keepdims=True)
            ).reshape(self.NP)
            Z_pred[i + 1, :, 1] = (
                Z_F_t + Z_F_t * (dW * LAMBDA_HAT_F_t).sum(axis=1, keepdims=True)
            ).reshape(self.NP)
            Q_pred[i + 1, :, 0] = (
                Q_D_t
                + Q_D_t
                * (
                    (
                        (R_D_t - R_F_t).reshape(self.NP, 1)
                        + np.sum(
                            (THETA_D_t - THETA_F_t) * THETA_D_t, axis=1, keepdims=True
                        )
                    )
                    * self.dt
                    + np.sum((THETA_D_t - THETA_F_t) * dW, axis=1, keepdims=True)
                )
            ).reshape(self.NP)

            Y_1_max = np.maximum(Y_pred[i + 1, :, 0].copy().reshape(self.NP, 1), 0)
            Y_2_max = np.maximum(Y_pred[i + 1, :, 1].copy().reshape(self.NP, 1), 0)
            Y_3_max = np.maximum(Y_pred[i + 1, :, 2].copy().reshape(self.NP, 1), 0)
            Y_4_max = np.maximum(Y_pred[i + 1, :, 3].copy().reshape(self.NP, 1), 0)
            Y_1_min = np.minimum(Y_pred[i + 1, :, 0].copy().reshape(self.NP, 1), 0)
            Y_2_min = np.minimum(Y_pred[i + 1, :, 1].copy().reshape(self.NP, 1), 0)
            Y_3_min = np.minimum(Y_pred[i + 1, :, 2].copy().reshape(self.NP, 1), 0)
            Y_4_min = np.minimum(Y_pred[i + 1, :, 3].copy().reshape(self.NP, 1), 0)
            a_d = self.ad_1 * Y_1_max + self.ad_2 * Y_2_min + self.ad_3 * Y_3_max
            b_d = self.bd_1 * Y_1_min + self.bd_2 * Y_2_min + self.bd_4 * Y_4_min
            a_f = self.af_1 * Y_1_min + self.af_2 * Y_2_max + self.af_3 * Y_3_min
            b_f = self.bf_1 * Y_1_max + self.bf_2 * Y_2_max + self.bf_4 * Y_4_max

            LAMBDA_HAT_D_pred[i + 1, :, :] = (
                a_d @ self.SIGMA_DELTA_D.T + b_d @ self.SIGMA_DELTA_F.T + self.EPSILON
            )
            LAMBDA_HAT_F_pred[i + 1, :, :] = (
                a_f @ self.SIGMA_DELTA_D.T + b_f @ self.SIGMA_DELTA_F.T
            )

        return S_filt, R_filt, Q_filt, S_D_0, S_F_0, R_D_0, R_F_0, llh, param_df
