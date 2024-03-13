#include "ywy_flvis_kinetic_math.hpp"
#include <deque>
#include <mutex>

#include <cmath>
#include <iostream>
#include <vector>

#include <boost/bind.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "../mtk/types/vect.hpp"
#include "../mtk/types/SOn.hpp"
#include "../mtk/types/S2.hpp"
#include "../mtk/startIdx.hpp"
#include "../mtk/build_manifold.hpp"
#include "util.hpp"
#include <vector>
#include <cmath>





struct ywy_Quaternion {
    double w, x, y, z;
};

void shengyang_madgwick_filter(const std::vector<double>& accel_data, 
                const std::vector<double>& gyro_data, 
                const std::vector<double>& ba, 
                const std::vector<double>& bg, 
                double dt, double beta, double zeta, 
                ywy_Quaternion& q) {
    // Initialize quaternion (assuming initial orientation is identity)
    double qw = q.w, qx = q.x, qy = q.y, qz = q.z;

    // Gyroscope measurements
    double gx = gyro_data[0] - bg[0];
    double gy = gyro_data[1] - bg[1];
    double gz = gyro_data[2] - bg[2];

    // Accelerometer measurements (normalize)
    double ax = accel_data[0] - ba[0];
    double ay = accel_data[1] - ba[1];
    double az = accel_data[2] - ba[2];

    Eigen::Quaterniond omega(0,gx,gy,gz);
    Eigen::Quaterniond q_prev(qw,qx,qy,qz);
    Eigen::Quaterniond qdot = scalar_multi_q(0.5,q1_multi_q2(q_prev,omega));

    if((std::sqrt(ax*ax+ay*ay+az*az) - 9.8)<0.3){
        double norm = std::sqrt(ax * ax + ay * ay + az * az);
        ax /= norm;
        ay /= norm;
        az /= norm;

        Vec4 s;
        s[0] = 2*qx*(ay + 2*qw*qx + 2*qy*qz) - 2*qy*(ax - 2*qw*qy + 2*qx*qz);
        s[1] = 2*qw*(ay + 2*qw*qx + 2*qy*qz) + 2*qz*(ax - 2*qw*qy + 2*qx*qz) - 4*qx*(- 2*qx*qx - 2*qy*qy + az + 1);
        s[2] = 2*qz*(ay + 2*qw*qx + 2*qy*qz) - 2*qw*(ax - 2*qw*qy + 2*qx*qz) - 4*qy*(- 2*qx*qx - 2*qy*qy + az + 1);
        s[3] = 2*qx*(ax - 2*qw*qy + 2*qx*qz) + 2*qy*(ay + 2*qw*qx + 2*qy*qz);
        s*=s.norm();
        // Apply feedback step
        qdot.w() -= beta * s[0];
        qdot.x() -= beta * s[1];
        qdot.y() -= beta * s[2];
        qdot.z() -= beta * s[3];
    }
    Quaterniond q_new = q_plus_q(q_prev,scalar_multi_q(dt,qdot));
    q_new.normalize();
    

    // Update final quaternion
    q.w = q_new.w();
    q.x = q_new.x();
    q.y = q_new.y();
    q.z = q_new.z();
}



void ywy_madgwick_filter(const std::vector<double>& accel_data, 
                const std::vector<double>& gyro_data, 
                const std::vector<double>& ba, 
                const std::vector<double>& bg, 
                double dt, double beta, double zeta, 
                ywy_Quaternion& q) {
    // Initialize quaternion (assuming initial orientation is identity)
    double q0 = q.w, q1 = q.x, q2 = q.y, q3 = q.z;

    // Gyroscope measurements
    double gx = gyro_data[0] - bg[0];
    double gy = gyro_data[1] - bg[1];
    double gz = gyro_data[2] - bg[2];

    // Accelerometer measurements (normalize)
    double ax = accel_data[0] - ba[0];
    double ay = accel_data[1] - ba[1];
    double az = accel_data[2] - ba[2];
    double norm = std::sqrt(ax * ax + ay * ay + az * az);
    ax /= norm;
    ay /= norm;
    az /= norm;

    // Auxiliary variables
    double half_dt = 0.5 * dt;
    double half_beta = 0.5 * beta;

    // Compute quaternion rate of change
    // double qDot0 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
    // double qDot1 = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
    // double qDot2 = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
    // double qDot3 = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

    // // Update quaternion
    // q0 += qDot0 * dt;
    // q1 += qDot1 * dt;
    // q2 += qDot2 * dt;
    // q3 += qDot3 * dt;

    // Normalize quaternion
    norm = std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 /= norm;
    q1 /= norm;
    q2 /= norm;
    q3 /= norm;

    // Compute error between estimated and measured acceleration
    double f1 = 2.0 * (q1 * q3 - q0 * q2) - ax;
    double f2 = 2.0 * (q0 * q1 + q2 * q3) - ay;
    double f3 = 2.0 * (0.5 - q1 * q1 - q2 * q2) - az;

    // Compute gradient descent step
    double J1[4] = {-2*q2, 2*q3, -2*q0, 2*q1};
    double J2[4] = {2*q1, 2*q0, 2*q3, 2*q2};
    double J3[4] = {0, -4*q1, -4*q2, 0};
    double step0 = J1[0] * f1 + J2[0] * f2 + J3[0] * f3;
    double step1 = J1[1] * f1 + J2[1] * f2 + J3[1] * f3;
    double step2 = J1[2] * f1 + J2[2] * f2 + J3[2] * f3;
    double step3 = J1[3] * f1 + J2[3] * f2 + J3[3] * f3;

    // Update quaternion using gradient descent
    std::cout << "step1" << step1 << std::endl;
    if(!isnan(step0)){
        q0 -= half_beta * step0;
        q1 -= half_beta * step1;
        q2 -= half_beta * step2;
        q3 -= half_beta * step3;
    }
    

    // Normalize quaternion again
    norm = std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 /= norm;
    q1 /= norm;
    q2 /= norm;
    q3 /= norm;

    // Update final quaternion
    q.w = q0;
    q.x = q1;
    q.y = q2;
    q.z = q3;
}


void ywy_madgwick_filter_diverge(const std::vector<double>& accel_data, 
                const std::vector<double>& gyro_data, 
                double dt, double beta, 
                ywy_Quaternion& q) {
    double ax = accel_data[0];
    double ay = accel_data[1];
    double az = accel_data[2];
    double gx = gyro_data[0];
    double gy = gyro_data[1];
    double gz = gyro_data[2];

    double q0 = q.w, q1 = q.x, q2 = q.y, q3 = q.z;
    std::cout << "q input: q.w=" << q.w << "q.x=" << q.x << "q.y=" << q.y << "q.z=" << q.z << std::endl;
    

    // 预测四元数
    double q0_pred = q0 - 0.5 * (q1 * gx + q2 * gy + q3 * gz) * dt;
    double q1_pred = q1 + 0.5 * (q0 * gx + q2 * gz - q3 * gy) * dt;
    double q2_pred = q2 + 0.5 * (q0 * gy - q1 * gz + q3 * gx) * dt;
    double q3_pred = q3 + 0.5 * (q0 * gz + q1 * gy - q2 * gx) * dt;
    std::cout << "q0_pred" << q0_pred << std::endl;


    double accel_norm = std::sqrt(ax * ax + ay * ay + az * az);
    ax /= accel_norm;
    ay /= accel_norm;
    az /= accel_norm;

    // 计算四元数更新误差
    double f1 = 2.0 * (q1_pred * q3_pred - q0_pred * q2_pred) - ax;
    double f2 = 2.0 * (q0_pred * q1_pred + q2_pred * q3_pred) - ay;
    double f3 = 2.0 * (0.5 - q1_pred * q1_pred - q2_pred * q2_pred) - az;

    // 计算四元数更新误差对四元数各项分别的导数
    double df1_dq0 = -2.0 * q2_pred;
    double df1_dq1 = 2.0 * q3_pred;
    double df1_dq2 = -2.0 * q0_pred;
    double df1_dq3 = 2.0 * q1_pred;

    double df2_dq0 = 2.0 * q1_pred;
    double df2_dq1 = 2.0 * q0_pred;
    double df2_dq2 = 2.0 * q3_pred;
    double df2_dq3 = 2.0 * q2_pred;

    double df3_dq0 = 0.0;
    double df3_dq1 = -4.0 * q1_pred;
    double df3_dq2 = -4.0 * q2_pred;
    double df3_dq3 = 0.0;

    // 计算原始导数
    double q0_dot = 0.5 * (-q1 * gx - q2 * gy - q3 * gz);
    double q1_dot = 0.5 * (q0 * gx + q2 * gz - q3 * gy);
    double q2_dot = 0.5 * (q0 * gy - q1 * gz + q3 * gx);
    double q3_dot = 0.5 * (q0 * gz + q1 * gy - q2 * gx);

    // 计算Jacobian_transpose加权
    // 更新四元数
    ywy_Quaternion delta_f;
    delta_f.w = (df1_dq0 * f1 + df2_dq0 * f2 + df3_dq0 * f3);
    delta_f.x = (df1_dq1 * f1 + df2_dq1 * f2 + df3_dq1 * f3);
    delta_f.y = (df1_dq2 * f1 + df2_dq2 * f2 + df3_dq2 * f3);
    delta_f.z = (df1_dq3 * f1 + df2_dq3 * f2 + df3_dq3 * f3);
    // 归一化四元数 delta_f
    double delta_f_norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    delta_f.w /= delta_f_norm;
    delta_f.x /= delta_f_norm;
    delta_f.y /= delta_f_norm;
    delta_f.z /= delta_f_norm;
    std::cout << "delta_f.w" << delta_f.w << std::endl;


    // 更新四元数
    if(!isnan(delta_f.w)){
        q.w = q.w + q0_dot * dt + beta*delta_f.w;
        q.x = q.x + q1_dot * dt + beta*delta_f.x;
        q.y = q.y + q2_dot * dt + beta*delta_f.y;
        q.z = q.z + q3_dot * dt + beta*delta_f.z;
        // hello world
    }

    // 归一化四元数
    double q_norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    q.w /= q_norm;
    q.x /= q_norm;
    q.y /= q_norm;
    q.z /= q_norm;
    std::cout << "q by madgwick: q.w=" << q.w << "q.x=" << q.x << "q.y=" << q.y << "q.z=" << q.z << std::endl;
}