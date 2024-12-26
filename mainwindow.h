#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QAction>
#include <QDebug>
#include <QDialog>
#include <QMessageBox>
#include <QFileDialog>
#include <QImage>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <QButtonGroup>
#include <cstdlib>
#include <iostream>
#include <QIntValidator>
#include <cmath>
#include <QSlider>
#include <fstream>
using namespace cv;

using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    Mat image_raw,image_result,image_reduced;
    QImage image_qt,image_qt_result,image_qt_reduced;
    QString filename;
    float reduce_scale;//图片尺寸过大无法在画布上显示，故需要缩小，这是缩小的比例
    bool large=false;
    Mat gradXY,theta;
    Mat RGB2GRAY(Mat img);
    Mat RGB2HSV(Mat img);
    Mat Horizontal_Mirroring(Mat image);//水平镜像
    Mat Vertical_Mirroring(Mat image);//垂直镜像
    Mat Rotate(Mat image,float deg);//旋转
    Mat Threshold_Segmentation(Mat image,int threshold);//阈值分割
    Mat Reverse(Mat image);//反向
    Mat Median_Filtering(Mat image,int kernal);//中值滤波
    Mat Mean_Filtering(Mat image,int kernal);//均值滤波
    Mat Gaussian_Filtering(Mat image,int kernal);//高斯滤波
    Mat Sobel_Filtering(Mat image,int kernal);//sobel滤波
    Mat Laplace_Filtering(Mat image);//laplace滤波
    void Canny_Filtering(Mat img);//canny滤波
    Mat Erosion(Mat image);//腐蚀
    Mat Dilation(Mat image);//膨胀
    Mat Histogram_Equalization(Mat img);//直方图均衡
    Mat Scaling(Mat image,float scale);//缩放
    void dajin(Mat image);//缩放
    void getGrandient (Mat img);
    Mat nonLocalMaxValue (Mat gradXY, Mat theta);
    Mat doubleThreshold (Mat image);
private slots:
    void on_image_trans_clicked();

    void on_image_filter_clicked();

    void on_start_clicked();

    void on_image_raw_clicked();

    void on_image_result_clicked();

    void openimage();

    bool check(QImage image_qt);

    void on_gray_clicked();

    void on_hsv_clicked();

    void on_action1_triggered();
    void on_action2_triggered();
    void on_action3_triggered();
    void on_action4_triggered();
    void on_action5_triggered();
    void on_action6_triggered();
    void on_action7_triggered();
    void on_action8_triggered();
    void on_action9_triggered();
    void on_action10_triggered();
    void on_actionstart_triggered();

    void on_horizon_clicked();

    void on_vertical_clicked();

    void on_reverse_2_clicked();

    void on_default_segmentation_clicked();

    void on_color_segmentation_clicked();

    void on_mean_3_clicked();

    void on_mean_5_clicked();

    void on_mean_7_clicked();

    void on_mean_12_clicked();

    void on_mean_13_clicked();

    void on_noise_clicked();

    void on_mean_9_clicked();

    void on_mean_10_clicked();

    void on_mean_11_clicked();

    void on_mean_15_clicked();

    void on_mean_16_clicked();

    void on_mean_18_clicked();

    void on_reverse_gray_clicked();

    void on_save_result_clicked();

    void on_Erosion_2_clicked();

    void on_Dilation_2_clicked();

    void on_open_clicked();

    void on_close_clicked();   

    void on_reduce_laplace_clicked();

    void on_abs_clicked();

    void on_histograam_clicked();

    void on_hist_color_clicked();

    void on_horizontalSlider_valueChanged(int value);

    void on_dajin_clicked();

    void on_radioButton_2_clicked();

    void on_radioButton_clicked();

    void on_rotate_slider_valueChanged(int value);

private:
    Ui::MainWindow *ui;
};


#endif // MAINWINDOW_H
