#include "mainwindow.h"
#include "ui_mainwindow.h"
//opencv 图片通道默认BGR, Qimage通道默认为 RGB
cv::Mat QImage2cvMat(QImage &image, bool rb_swap)
{
    cv::Mat mat;
    switch(image.format())
    {
        case QImage::Format_RGB888:
            mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void *)image.constBits(), image.bytesPerLine());
            mat = mat.clone();
            if(rb_swap) cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
            break;
        case QImage::Format_Indexed8:
        case QImage::Format_Grayscale8:
            mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void *)image.bits(), image.bytesPerLine());
            mat = mat.clone();
            break;
        case QImage::Format_ARGB32:
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32_Premultiplied:
            auto mat_tmp = cv::Mat(image.height(), image.width(), CV_8UC4, (void *)image.constBits(), image.bytesPerLine());
            auto img=cv::Mat(image.height(), image.width(), CV_8UC3, cv::Scalar(0));
            cv::cvtColor(mat_tmp , img , cv::COLOR_RGBA2RGB);
            mat = img.clone();
            break;
    }
    return mat;
}

//opencv图片转换成qt图片
QImage cvMat2QImage(const cv::Mat& mat, bool rb_swap)
{
    const uchar *pSrc = (const uchar*)mat.data;
    if(mat.type() == CV_8UC1)
    {
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        return image.copy();

    }
    else if(mat.type() == CV_8UC3)
    {
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        if(rb_swap) return image.rgbSwapped();
        return image.copy();
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}


Mat addSaltNoise(const Mat srcImage, int n)
{
    Mat dstImage = srcImage.clone();//为什么要克隆一下，image.at<uchar>是操作指针，会改变原图
    for (int k = 0; k < n; k++)
    {
        //随机取值行列
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        //图像通道判定
        if (dstImage.channels() == 1)
        {
            dstImage.at<uchar>(i, j) = 255;		//盐噪声
        }
        else
        {
            dstImage.at<Vec3b>(i, j)[0] = 255;
            dstImage.at<Vec3b>(i, j)[1] = 255;
            dstImage.at<Vec3b>(i, j)[2] = 255;
        }
    }
    for (int k = 0; k < n; k++)
    {
        //随机取值行列
        int i = rand() % dstImage.rows;
        int j = rand() % dstImage.cols;
        //图像通道判定
        if (dstImage.channels() == 1)
        {
            dstImage.at<uchar>(i, j) = 0;		//椒噪声
        }
        else
        {
            dstImage.at<Vec3b>(i, j)[0] = 0;
            dstImage.at<Vec3b>(i, j)[1] = 0;
            dstImage.at<Vec3b>(i, j)[2] = 0;
        }
    }
    return dstImage;
}

//卷积，手写卷积，没有调用库函数，只支持灰度图，彩色图还没写
Mat colcon(Mat image,vector<vector<float>> kernal)
{
    Mat img;

    if(image.channels()==1)img=cv::Mat(image.rows,image.cols,CV_8UC1,cv::Scalar(0));
    else  img=cv::Mat(image.rows,image.cols,CV_8UC3,cv::Scalar(0));
    int width = img.cols;		// 获取图像宽度
    int height = img.rows;	// 获取图像高度S
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
        {
            if(kernal.size()==3)
            {
                float tmp=0.0;
                for(int i=-1;i<=1;i++)
                    for(int j=-1;j<=1;j++)
                    {
                        if(row+i>=0&&row+i<height&&col+j>=0&&col+j<width)
                        tmp+=kernal[i+1][j+1]*(float)image.at<uchar>(row+i,col+j);
                    }
//                 img.at<uchar>(row,col)=(int)(abs(tmp));
//                 qDebug()<<img.at<uchar>(row,col);
                img.at<uchar>(row,col)=(tmp>0)?(int)tmp:0;//小于零全部抹除，可以用上面的绝对值替换
            }
            else if(kernal.size()==5)
            {
                float tmp=0.0;
                for(int i=-2;i<=2;i++)
                    for(int j=-2;j<=2;j++)
                    {
                        if(row+i<0||row+i>=height||col+j<0||col+j>=width)continue;
                        tmp+=kernal[i+2][j+2]*image.at<uchar>(row+i,col+j);
                    }
                // img.at<uchar>(row,col)=(int)(abs(tmp));
                img.at<uchar>(row,col)=(tmp>0)?(int)tmp:0;
            }
            else
            {
                float tmp=0.0;
                for(int i=-3;i<=3;i++)
                    for(int j=-3;j<=3;j++)
                    {
                        if(row+i<0||row+i>=height||col+j<0||col+j>=width)continue;
                        tmp+=kernal[i+3][j+3]*image.at<uchar>(row+i,col+j);
                    }
                // img.at<uchar>(row,col)=(int)(abs(tmp));
                img.at<uchar>(row,col)=(tmp>0)?(int)tmp:0;
            }
        }
    return img;
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->action_O,&QAction::triggered,this,&MainWindow::openimage);
    QButtonGroup *buttonGroup = new QButtonGroup(this);
    buttonGroup->addButton(ui->gray);
    buttonGroup->addButton(ui->hsv);
    buttonGroup->addButton(ui->horizon);
    buttonGroup->addButton(ui->vertical);
    buttonGroup->addButton(ui->default_segmentation);
    buttonGroup->addButton(ui->dajin);
    buttonGroup->addButton(ui->reverse_2);
    buttonGroup->addButton(ui->Erosion_2);
    buttonGroup->addButton(ui->Dilation_2);
    buttonGroup->addButton(ui->histograam);
    buttonGroup->addButton(ui->color_segmentation);
    buttonGroup->addButton(ui->reverse_gray);
    buttonGroup->addButton(ui->open);
    buttonGroup->addButton(ui->close);
    buttonGroup->addButton(ui->hist_color);//带有颜色的直方图均衡
    QButtonGroup *buttonGroup2 = new QButtonGroup(this);
    //上面这些radiobutton只有一个能够同时被勾选，下面的是另一组
    buttonGroup2->addButton(ui->mean_3);
    buttonGroup2->addButton(ui->mean_5);
    buttonGroup2->addButton(ui->mean_7);
    buttonGroup2->addButton(ui->mean_9);
    buttonGroup2->addButton(ui->mean_10);
    buttonGroup2->addButton(ui->mean_11);
    buttonGroup2->addButton(ui->mean_12);
    buttonGroup2->addButton(ui->mean_13);
    buttonGroup2->addButton(ui->mean_15);
    buttonGroup2->addButton(ui->mean_16);
    buttonGroup2->addButton(ui->mean_18);
    buttonGroup2->addButton(ui->radioButton);
    buttonGroup2->addButton(ui->reduce_laplace);//锐化
    buttonGroup2->addButton(ui->abs);//锐化
    buttonGroup2->addButton(ui->radioButton_2);//锐化


    filename=":/new/prefix1/start.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==true)
    {
        qDebug()<<"加载了一张图片";
    }
    // 设置滑动条的最小值和最大值

    ui->horizontalSlider->setMinimum(10);
    ui->horizontalSlider->setMaximum(150);

    // 设置滑动条的单步值（每个刻度的变化量）
    ui->horizontalSlider->setSingleStep(1);

    // 设置滑动条的刻度数量
    ui->horizontalSlider->setTickInterval(1); // 每个刻度之间的值差
    ui->horizontalSlider->setValue(100);
    ui->horizontalSlider->setTickPosition(QSlider::TicksBelow); // 刻度显示在滑动条下方


    ui->rotate_slider->setMinimum(-90);
    ui->rotate_slider->setMaximum(90);
    // 设置滑动条的单步值（每个刻度的变化量）
    ui->rotate_slider->setSingleStep(1);
    // 设置滑动条的刻度数量
    ui->rotate_slider->setTickInterval(1); // 每个刻度之间的值差
    ui->rotate_slider->setValue(0);


    ui->rotate_slider->setTickPosition(QSlider::TicksBelow); // 刻度显示在滑动条下方
    ui->label->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    ui->label_2->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    ui->label_3->setAlignment(Qt::AlignTop | Qt::AlignLeft);
}
MainWindow::~MainWindow()
{
    delete ui;
}

//false 没有超限，函数目的检查图片大小是否超过显示范围，超过就适当缩小图片便于显示
bool MainWindow::check(QImage image_qt)
{
    if(image_qt.width()>750||image_qt.height()>520)
    {
        large=true;
        if((float)image_qt.width()/(float)image_qt.height()>=750.0/520.0)
            reduce_scale=750/(float)image_qt.width();
        else
            reduce_scale=520/(float)image_qt.height();
        image_qt_reduced=image_qt.scaled((int)(image_qt.width()*reduce_scale-1),
                                     (int)(image_qt.height()*reduce_scale-1), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        image_raw=QImage2cvMat(image_qt_reduced,false);
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt_reduced));
        ui->stackedWidget->setCurrentIndex(1);
        return true;
    }
    image_raw=QImage2cvMat(image_qt,false);
    return false;
}
void MainWindow::openimage()
{
    filename=QFileDialog::getOpenFileName(this,"选择图片",QCoreApplication::applicationDirPath(),
                                                       "Images (*.png *.bmp *.jpg *.jpeg)");
    if(filename.isEmpty())
    {
        return;
    }
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
}

// 转为灰度图
Mat MainWindow::RGB2GRAY(Mat img)
{
    int width = img.cols;		// 获取图像宽度
    int height = img.rows;	// 获取图像高度
    cv::Mat grayImage(height, width, CV_8UC1, cv::Scalar(0));
    if(img.channels()==1)   return img.clone();
    else
    {
        // qDebug()<<"图片宽度"<<img.cols<<img.rows<<"\n";
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                Vec3b bgr = img.at<Vec3b>(row, col);
                grayImage.at<uchar>(row, col)=0.1140*bgr[0]+0.5870*bgr[1]+0.2989*bgr[2];
            }
        }
    }
    return grayImage;
};

// 转为hsv
Mat MainWindow::RGB2HSV(Mat img)
{
    auto hsvImage = cv::Mat(img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
        {
            auto &rgb = img.at<Vec3b>(i, j);
            float r = rgb[2]/255.0, g = rgb[1]/255.0, b = rgb[0]/255.0;
            float h, s, v;
            float minVal = std::min({r, g, b});
            float maxVal = std::max({r, g, b});
            v = maxVal; // value
            s=(maxVal==0)?0:(1-minVal/maxVal);
            float delta = maxVal - minVal;
            if (delta == 0) h = 0; // undefined hue
            else
            {
                if (maxVal == r&&g>=b)
                    h =60*(g-b)/(maxVal - minVal);
                else if (maxVal == r&&g<b)
                    h =60*(g-b)/(maxVal - minVal)+360;
                else if (maxVal == g)
                    h =120+60*(b-r)/(maxVal - minVal);
                else
                    h =240+60*(r-g)/(maxVal - minVal);
            }
            hsvImage.at<Vec3b>(i, j)[0] = (int)(h*255/360);
            hsvImage.at<Vec3b>(i, j)[1] = (int)(s*255);
            hsvImage.at<Vec3b>(i, j)[2] = (int)(v*255);

        }
    return hsvImage;
};

Mat MainWindow::Horizontal_Mirroring(Mat image)
{
    auto img=image.clone();
    int width = img.cols;		// 获取图像宽度
    int height = img.rows;	// 获取图像高度
    int channels = img.channels();	// 获取图像通道数
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
        {
            if (channels == 1)	//单通道，图像为灰度
                img.at<uchar>(row, width-col-1) = image.at<uchar>(row, col);
            if (channels == 3) //三通道图像，彩色图像 BGR
                img.at<Vec3b>(row, width-col-1)= image.at<Vec3b>(row, col);
        }
    return img;
};//水平镜像

Mat MainWindow::Vertical_Mirroring(Mat image)
{
    auto img=image.clone();
    int width = img.cols;		// 获取图像宽度
	int height = img.rows;	// 获取图像高度
	int channels = img.channels();	// 获取图像通道数
	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
		{
			if (channels == 1)	//单通道，图像为灰度
                img.at<uchar>(height-row-1,col) =image.at<uchar>(row, col);
			if (channels == 3) //三通道图像，彩色图像 BGR
                img.at<Vec3b>(height-row-1, col)  = image.at<Vec3b>(row, col);
		}
    return img;
};//垂直镜像

//一个点绕中心旋转角度，返回值，返回旋转后的坐标 P'=R*(P-C)+C
vector<float> rotate(vector<int> point,vector<int> center,float degree)
{
    vector<float> result(2),temp(2);//结果和要旋转的点到中心距离的向量
    temp[0]=(float)(point[0]-center[0]);//(P-C)
    temp[1]=(float)(point[1]-center[1]);
    auto rad=degree*M_PI/180.0;
    result[0]=cos(rad)*temp[0]-sin(rad)*temp[1]+(float)center[0];
    result[1]=sin(rad)*temp[0]+cos(rad)*temp[1]+(float)center[1];
    return result;//旋转之后的坐标
}

//旋转后的坐标求解旋转前的坐标，返回旋转之前的坐标 R^{-1}*(P'-C)+C=P
vector<float> rotate_inv(vector<int> point,vector<int> center,float degree)
{
    vector<float> result(2),temp(2);//结果和要旋转的点到中心距离的向量
    temp[0]=(float)(point[0]-center[0]);//(P'-C)
    temp[1]=(float)(point[1]-center[1]);
    auto rad=degree*M_PI/180.0;
    result[0]=cos(rad)*temp[0]+sin(rad)*temp[1]+(float)center[0];
    result[1]=-sin(rad)*temp[0]+cos(rad)*temp[1]+(float)center[1];
    return result;//旋转之前的坐标
}

Mat MainWindow::Rotate(Mat image,float deg)
{
    auto width=image.cols;
    auto height=image.rows;
    vector<int> left_up(2),left_down(2),right_up(2),right_down(2);
    left_up={0,0},left_down={height,0},right_up={0,width},right_down={height,width};
    auto rotated_left_up=rotate(left_up,left_down,deg),rotated_left_down=rotate(left_down,left_down,deg),
    rotated_right_up=rotate(right_up,left_down,deg),rotated_right_down=rotate(right_down,left_down,deg);
    int h=(int)std::max({rotated_left_up[0],rotated_left_down[0],rotated_right_up[0],rotated_right_down[0]})-
    (int)std::min({rotated_left_up[0],rotated_left_down[0],rotated_right_up[0],rotated_right_down[0]});
    int w=(int)std::max({rotated_left_up[1],rotated_left_down[1],rotated_right_up[1],rotated_right_down[1]})-
    (int)std::min({rotated_left_up[1],rotated_left_down[1],rotated_right_up[1],rotated_right_down[1]});
    cv::Mat img;
    int min_h=(int)std::min({rotated_left_up[0],rotated_left_down[0],rotated_right_up[0],rotated_right_down[0]});
    int min_w=(int)std::min({rotated_left_up[1],rotated_left_down[1],rotated_right_up[1],rotated_right_down[1]});
    if(image.channels()==1) 
    {
        img=cv::Mat(h,w,CV_8UC1,cv::Scalar(0));
        for (int i=0;i<h;i++)
            for(int j=0;j<w;j++)//对平移后的像素操作
            {
                vector<int> before_trans(2);
                //原图旋转后的像素坐标有负值，图像上没有负值坐标，故应进行一次平移变换，平移（-min），这样负值加上（-min）
                //就变成了正值，下面是求解平移变换之前的坐标
                before_trans[0]=i+min_h;
                before_trans[1]=j+min_w;
                auto before_rotate=rotate_inv(before_trans,left_down,deg);
                if(before_rotate[0]<0||before_rotate[0]>=height||before_rotate[1]<0||before_rotate[1]>=width)
                img.at<uchar>(i,j)=0;
                else
                {
                    int floor_height=std::floor(before_rotate[0]),floor_width=std::floor(before_rotate[1]),
                    ceil_height=std::ceil(before_rotate[0]),ceil_width=std::ceil(before_rotate[1]);
                    float upper_rate=(float)ceil_height-before_rotate[0],low_rate=1-upper_rate,left_rate=(float)ceil_width-before_rotate[1],right_rate=1-left_rate;
                    img.at<uchar>(i,j)=(int)(upper_rate*(left_rate*image.at<uchar>(floor_height,floor_width)+right_rate*image.at<uchar>(floor_height,ceil_width))
                    +low_rate*(left_rate*image.at<uchar>(ceil_height,floor_width)+right_rate*image.at<uchar>(ceil_height,ceil_width)));
                }
            }
    }
    else if (image.channels()==3)
    {
        img=cv::Mat(h,w,CV_8UC3,cv::Scalar(0));
        for (int i=0;i<h;i++)
            for(int j=0;j<w;j++)//对平移后的像素操作
            {
                vector<int> before_trans(2);
                //原图旋转后的像素坐标有负值，图像上没有负值坐标，故应进行一次平移变换，平移（-min），这样负值加上（-min）
                //就变成了正值，下面是求解平移变换之前的坐标
                before_trans[0]=i+min_h;
                before_trans[1]=j+min_w;
                auto before_rotate=rotate_inv(before_trans,left_down,deg);
                if(before_rotate[0]>=0&&before_rotate[0]<height&&before_rotate[1]>=0&&before_rotate[1]<width)
                {
                    int floor_height=std::floor(before_rotate[0]),floor_width=std::floor(before_rotate[1]),
                    ceil_height=std::ceil(before_rotate[0]),ceil_width=std::ceil(before_rotate[1]);
                    float upper_rate=(float)ceil_height-before_rotate[0],low_rate=1-upper_rate,left_rate=(float)ceil_width-before_rotate[1],right_rate=1-left_rate;
                    img.at<Vec3b>(i,j)=(Vec3b)(upper_rate*(left_rate*image.at<Vec3b>(floor_height,floor_width)+right_rate*image.at<Vec3b>(floor_height,ceil_width))
                    +low_rate*(left_rate*image.at<Vec3b>(ceil_height,floor_width)+right_rate*image.at<Vec3b>(ceil_height,ceil_width)));
                }
            }
    }
    return img;

}


Mat MainWindow::Threshold_Segmentation(Mat image,int threshold)
{
    auto img=image.clone();
    int width = img.cols;		// 获取图像宽度
	int height = img.rows;	// 获取图像高度
	int channels = img.channels();	// 获取图像通道数
	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
		{
			if (channels == 1)	//单通道，图像为灰度
			{
				int pv = img.at<uchar>(row, col);
                if(pv>=threshold) img.at<uchar>(row, col) = 255;
                else img.at<uchar>(row, col) = 0;
			}
			if (channels == 3) //三通道图像，彩色图像 BGR
			{
				Vec3b bgr = img.at<Vec3b>(row, col);
				if(bgr[0]>=threshold) img.at<Vec3b>(row, col)[0] = 255;
                else img.at<Vec3b>(row, col)[0]  = 0;
                if(bgr[1]>=threshold) img.at<Vec3b>(row, col)[1] = 255;
                else img.at<Vec3b>(row, col)[1]  = 0;
                if(bgr[2]>=threshold) img.at<Vec3b>(row, col)[2] = 255;
                else img.at<Vec3b>(row, col)[2]  = 0;

			}
		}
    return img;

};//阈值分割

Mat MainWindow::Reverse(Mat image)
{
    auto img=image.clone();
    int width = img.cols;		// 获取图像宽度
	int height = img.rows;	// 获取图像高度
	int channels = img.channels();	// 获取图像通道数
	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
		{
			if (channels == 1)	//单通道，图像为灰度
			{
				int pv = img.at<uchar>(row, col);
				img.at<uchar>(row, col) = 255 - pv;
			}
			if (channels == 3) //三通道图像，彩色图像 BGR
			{
				Vec3b bgr = img.at<Vec3b>(row, col);
				img.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				img.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				img.at<Vec3b>(row, col)[2] = 255 - bgr[2];
			}
		}
    return img;
};//反向

Mat MainWindow::Median_Filtering(Mat image,int kernal)
{
    auto img=image.clone();
    int width = img.cols;		// 获取图像宽度
    int height = img.rows;	// 获取图像高度
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
        {
            if(kernal==3)
            {
                vector<int> vec;
                for(int i=-1;i<=1;i++)
                    for(int j=-1;j<=1;j++)
                    {
                        if(row+i<0||row+i>=height||col+j<0||col+j>=width)continue;
                        vec.push_back(image.at<uchar>(row+i,col+j));
                    }
                std::sort(vec.begin(), vec.end());
                img.at<uchar>(row,col)=vec[(int)(vec.size()/2)];
            }
            else if(kernal==5)
            {
                vector<int> vec;
                for(int i=-2;i<=2;i++)
                    for(int j=-2;j<=2;j++)
                    {
                        if(row+i<0||row+i>=height||col+j<0||col+j>=width)continue;
                        vec.push_back(image.at<uchar>(row+i,col+j));
                    }
                std::sort(vec.begin(), vec.end());
                img.at<uchar>(row,col)=vec[(int)(vec.size()/2)];
            }
            else
            {
                vector<int> vec;
                for(int i=-3;i<=3;i++)
                    for(int j=-3;j<=3;j++)
                    {
                        if(row+i<0||row+i>=height||col+j<0||col+j>=width)continue;
                        vec.push_back(image.at<uchar>(row+i,col+j));
                    }
                std::sort(vec.begin(), vec.end());
                img.at<uchar>(row,col)=vec[(int)(vec.size()/2)];
            }
        }
    return img;
};//中值滤波

Mat MainWindow::Mean_Filtering(Mat image,int kernal)
{
    auto img=image.clone();
    if(kernal==3)
    {
        vector<vector<float>> kernel = {{1.0/9.0, 1.0/9.0, 1.0/9.0},
                             {1.0/9.0, 1.0/9.0, 1.0/9.0},
                             {1.0/9.0, 1.0/9.0, 1.0/9.0}};
        return colcon(image,kernel);
    }
    else if(kernal==5)
    {
        vector<vector<float>> kernel = {
            {1.0/25.0, 1.0/25, 1.0/25, 1.0/25, 1.0/25},
            {1.0/25, 1.0/25, 1.0/25, 1.0/25, 1.0/25},
            {1.0/25, 1.0/25, 1.0/25, 1.0/25, 1.0/25},
            {1.0/25, 1.0/25, 1.0/25, 1.0/25, 1.0/25},
            {1.0/25, 1.0/25, 1.0/25, 1.0/25, 1.0/25}
        };
        return colcon(image,kernel);
    }
    else
    {
        std::vector<std::vector<float>> kernel = {
            {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
            {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
            {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
            {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
            {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
            {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0},
            {1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0, 1.0/49.0}};
        return colcon(image,kernel);
    }
};//均值滤波

Mat MainWindow::Gaussian_Filtering(Mat image,int kernal)
{
    auto img=image.clone();
    if(kernal==3)
    {
        cv::Mat kernel = (cv::Mat_<float>(3,3) <<
            0.05, 0.15, 0.05,
            0.15, 0.2, 0.15,
            0.05, 0.15, 0.05);
        cv::Mat result;
        cv::filter2D(img, result, -1, kernel);
        return result;
    }
    else if(kernal==5)
    {
        cv::Mat kernel = (cv::Mat_<float>(5,5) <<
            1,2,4,2,1,
            2,4,16,4,2,
            4,16,64,16,4,
            2,4,16,4,2,
            1,2,4,2,1)/180.0;
        cv::Mat result;
        cv::filter2D(img, result, -1, kernel);
        return result;
    }
    return img;
};//高斯滤波

Mat MainWindow::Sobel_Filtering(Mat image,int kernal)
{
    auto img=image.clone();
    if(kernal==0)//X方向
    {
//         cv::Mat kernel = (cv::Mat_<int>(3,3) <<
//             -1, 0, 1,
//             -2, 0, 2,
//             -1, 0, 1);
//         cv::Mat result;
//         cv::filter2D(img, result, -1, kernel);
//         return result;

         vector<vector<float>> kernel = {
                              {-1.0, 0.0, 1.0},
                              {-2.0,0.0, 2.0},
                              {-1.0, 0.0, 1.0}};
         return colcon(image,kernel);

    }
    else//Y方向
    {
         cv::Mat kernel = (cv::Mat_<int>(3,3) <<
             -1, -2, -1,
             0, 0, 0,
             1, 2, 1);
         cv::Mat result;
         cv::filter2D(img, result, -1, kernel);
         return result;
    }
    return img;
};//sobel滤波

//Mat MainWindow::Laplace_Filtering(Mat image)
Mat MainWindow::Laplace_Filtering(Mat image)
{
    auto img=image.clone();
    vector<vector<float>> kernel = {
                         {0.0, 1.0, 0.0},
                         {1.0,-4.0, 1.0},
                         {0.0, 1.0,0.0}};
    return colcon(image,kernel);
//    cv::Mat kernel = (cv::Mat_<int>(3,3) <<
//        0, 1, 0,
//        1, -4, 1,
//        0, 1, 0);
//    cv::Mat result;
//    cv::filter2D(img, result, -1, kernel);
//    return result;
};//laplace滤波，滤波相当于对图像进行二阶求导，如果用原图减去Laplace,锐化处理

//求图片的梯度和模长
void MainWindow::getGrandient (Mat img)
{

    gradXY = Mat(img.rows,img.cols,CV_8UC1,cv::Scalar(0));
    theta=Mat(img.rows,img.cols,CV_32FC1,cv::Scalar(0));
    for (int i = 1; i < img.rows-1; i++) {
        for (int j = 1; j < img.cols-1; j++) {
            float gradX = float(-img.at<uchar>(i-1,j-1) - 2 * img.at<uchar>(i-1,j) - img.at<uchar>(i-1,j+1) + img.at<uchar>(i+1,j-1) + 2 * img.at<uchar>(i+1,j) + img.at<uchar>(i+1,j+1));
            float gradY = float(img.at<uchar>(i-1,j+1) + 2 * img.at<uchar>(i,j+1) + img.at<uchar>(i+1,j+1) - img.at<uchar>(i-1,j-1) - 2 * img.at<uchar>(i,j-1) - img.at<uchar>(i+1,j-1));
            gradXY.at<uchar>(i,j)= sqrt(gradY*gradY+gradX*gradX); //计算梯度
//            gradXY.at<uchar>(i,j)= abs(gradY)+abs(gradX); //计算梯度
            theta.at<float>(i,j)= atan2(gradY,gradX); //计算梯度方向
        }
    }
}

//四个方向的非极大值抑制
Mat MainWindow::nonLocalMaxValue (Mat gradXY, Mat theta) {
    auto img = gradXY.clone();
    for (int i = 1; i < gradXY.rows-1; i++) 
        for (int j = 1; j < gradXY.cols-1; j++) {
            float t = float(theta.at<uchar>(i,j));
            float g = float(gradXY.at<uchar>(i,j));
            if (g == 0.0) continue;
            double g0, g1;
            if ((t >= -(3*M_PI/8)) && (t < -(M_PI/8))) {
                g0 = double(gradXY.at<uchar>(i-1,j-1));
                g1 = double(gradXY.at<uchar>(i+1,j+1));
            }
            else if ((t >= -(M_PI/8)) && (t < M_PI/8)) {
                g0 = double(gradXY.at<uchar>(i,j-1));
                g1 = double(gradXY.at<uchar>(i,j+1));
            }
            else if ((t >= M_PI/8) && (t < 3*M_PI/8)) {
                g0 = double(gradXY.at<uchar>(i-1,j+1));
                g1 = double(gradXY.at<uchar>(i+1,j-1));
            }
            else {
                g0 = double(gradXY.at<uchar>(i-1,j));
                g1 = double(gradXY.at<uchar>(i+1,j));
            }
            
            if (g <= g0 || g <= g1) {
                img.at<uchar>(i,j) = 0;
            }
        }
    return img;
}

//双边缘阈值分割
Mat MainWindow::doubleThreshold (Mat image) {
    auto img = image.clone();
    double minValue, maxValue;
    cv::Point minIdx, maxIdx;
    // 寻找最大值和最小值
    cv::minMaxLoc(image, &minValue, &maxValue, &minIdx, &maxIdx);
    // 区分出弱边缘点和强边缘点
    qDebug()<<"最大值"<<maxValue;
    for (int i = 0; i < img.rows; i++) 
        for (int j = 0; j < img.cols; j++) 
        {
            if(image.at<uchar>(i,j)>=(int)(maxValue/4))img.at<uchar>(i,j)=255;
            else if(image.at<uchar>(i,j)<(int)(maxValue/6))img.at<uchar>(i,j)=0;
            else
            {
                for(int m=-1;m<=1;m++)
                    for(int n=-1;n<=1;n++)
                    {
                        if(i+m<0||i+m>=image.rows||j+n<0||j+n>=image.cols) continue;
                        else {
                            if(image.at<uchar>(i+m,j+n)>=(maxValue/4))img.at<uchar>(i,j)=255;
                        }
                    }
            }
        }

    return img;
}

//canny -> 高斯模糊 ->梯度检测-> 非极大值抑制->双边缘阈值分割 
void MainWindow::Canny_Filtering(Mat img)
{

    Mat tmp=RGB2GRAY(img);
    cv::Mat blurred_image;
    cv::GaussianBlur(tmp, blurred_image, cv::Size(5, 5), 0.2);
    getGrandient(blurred_image);
    auto tmep_2=nonLocalMaxValue(gradXY,theta);
    image_result=doubleThreshold(tmep_2);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);

};//canny滤波

Mat MainWindow::Erosion(Mat image)
{
    auto img=image.clone();
    int width = img.cols;		// 获取图像宽度
    int height = img.rows;	// 获取图像高度
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
        {
                float tmp=0.0;
                for(int i=-1;i<=1;i++)
                    for(int j=-1;j<=1;j++)
                    {
                        if(row+i<0||row+i>=height||col+j<0||col+j>=width)continue;
                        if(image.at<uchar>(row+i,col+j)==255)img.at<uchar>(row,col)=255;
                    }
        }
    return img;
};//腐蚀要对图像做二值化，腐蚀相对于黑色是腐蚀，那么相对于白色就是膨胀了

Mat MainWindow::Dilation(Mat image)
{
    auto img=image.clone();
    int width = img.cols;		// 获取图像宽度
    int height = img.rows;	// 获取图像高度
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
        {
                float tmp=0.0;
                for(int i=-1;i<=1;i++)
                    for(int j=-1;j<=1;j++)
                    {
                        if(row+i<0||row+i>=height||col+j<0||col+j>=width)continue;
                        if(image.at<uchar>(row+i,col+j)==0)img.at<uchar>(row,col)=0;
                    }
        }
    return img;
};//膨胀


Mat MainWindow::Histogram_Equalization(Mat image)
{
    auto img=image.clone();
    int width = img.cols;   // 获取图像宽度
    int height = img.rows;	// 获取图像高度
    if(image.channels()==1)
    {
        float hist[256]={0.0};
        for (int row = 0; row < height; row++)
            for (int col = 0; col < width; col++)
                hist[image.at<uchar>(row,col)]+=1.0;
        for(int n=1;n<=255;n++)hist[n]+=hist[n-1];
        for(int n=0;n<=255;n++)hist[n]*=255.0/(float)(width*height);
            //qDebug()<<(int)hist[n];
        for (int row = 0; row < height; row++)
            for (int col = 0; col < width; col++)
                img.at<uchar>(row,col)=(int)hist[image.at<uchar>(row,col)];
        }
    else if(image.channels()==3)
    {
        for(int i=0;i<3;i++)
        {
            float hist[256]={0.0};
            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++)
                    hist[image.at<Vec3b>(row,col)[i]]+=1.0;
            for(int n=1;n<=255;n++)hist[n]+=hist[n-1];
            for(int n=0;n<=255;n++)hist[n]*=255.0/(float)(width*height);
                //qDebug()<<(int)hist[n];
            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++)
                    img.at<Vec3b>(row,col)[i]=(int)hist[image.at<Vec3b>(row,col)[i]];
        }
    }
    return img;
};//直方图均衡

//图片缩放的双线性插值
Mat MainWindow::Scaling(Mat image,float scale)
{
    int width = image.cols;		// 获取图像宽度
    int height = image.rows;	// 获取图像高度
    cv::Mat img;
    int scaled_width=width*scale,scaled_height=scale*height;
    if(image.channels()==1) img=cv::Mat(scaled_height,scaled_width,CV_8UC1,cv::Scalar(0));
    else if (image.channels()==3)img=cv::Mat(scaled_height,scaled_width,CV_8UC3,cv::Scalar(0));
    for(int i=0;i<scaled_height;i++)
        for(int j=0;j<scaled_width;j++)
        {
            float height_returned,width_returned;
            // qDebug()<<height_returned<<" "<<width_returned;
            if((i+1)/scale-1>0) height_returned= (float)((i+1)/scale-1) ; else height_returned   =0.0;
            if((j+1)/scale-1>0) width_returned = (float)((j+1)/scale-1) ; else width_returned    =0.0;
            if(image.channels()==1)
            {
                int floor_height=std::floor(height_returned),floor_width=std::floor(width_returned),
                ceil_height=std::ceil(height_returned),ceil_width=std::ceil(width_returned);
                float upper_rate=(float)ceil_height-height_returned,low_rate=1-upper_rate,left_rate=(float)ceil_width-width_returned,right_rate=1-left_rate;
                img.at<uchar>(i,j)=(int)(upper_rate*(left_rate*image.at<uchar>(floor_height,floor_width)+right_rate*image.at<uchar>(floor_height,ceil_width))
                +low_rate*(left_rate*image.at<uchar>(ceil_height,floor_width)+right_rate*image.at<uchar>(ceil_height,ceil_width)));
                //qDebug()<<height_returned<<" "<<width_returned;
            }
            else if(image.channels()==3)
            {
                int floor_height=std::floor(height_returned),floor_width=std::floor(width_returned),
                ceil_height=std::ceil(height_returned),ceil_width=std::ceil(width_returned);
                float upper_rate=(float)ceil_height-height_returned,low_rate=1-upper_rate,left_rate=(float)ceil_width-width_returned,right_rate=1-left_rate;
                // for(int channel=0;channel<3;channel++)[channel]
                img.at<Vec3b>(i,j)=(Vec3b)(upper_rate*(left_rate*image.at<Vec3b>(floor_height,floor_width)+right_rate*image.at<Vec3b>(floor_height,ceil_width))
                +low_rate*(left_rate*image.at<Vec3b>(ceil_height,floor_width)+right_rate*image.at<Vec3b>(ceil_height,ceil_width)));
            }
        }
    return img;
};//缩放

void MainWindow::dajin(Mat image)//大津阈值分割
{
    int width = image.cols;		// 获取图像宽度
    int height = image.rows;	// 获取图像高度

    vector<int> mean_temp(256,0);//小于阈值的值
    vector<int> p(256,0);//小于阈值的个数
    vector<float> pp(256,0);//以索引值作阈值的小于阈值的像素点占所有像素点的概率
    vector<float> variance_temp(256,0.0);//每一个阈值下的方差
    int mean_all=0;
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
        {    
            for (int i=255;i>=image.at<uchar>(row,col);i--)//这个像素点，只有在以比他大的的值为阈值计算中才会被算到
            {
                mean_temp[i]+=image.at<uchar>(row,col);
                p[i]+=1;
            }
        }
    // qDebug()<<"到此没有bug";
    mean_all=mean_temp[255]/p[255];//总像素点除以所有像素点
    for(int i=1;i<255;i++)
    {
        if(p[i]==0){//防止除数出现零
            variance_temp[i]=-9999;
            continue;
        }
        variance_temp[i]=p[i]*(mean_all-mean_temp[i]/p[i])*(mean_all-mean_temp[i]/p[i])/(width*height-p[i]);
    }
    // qDebug()<<"到此没有bug2";
    // 使用std::max_element找到最大值的迭代器
    auto max_iter = std::max_element(variance_temp.begin(), variance_temp.end());
    int max_index = (int)std::distance(variance_temp.begin(), max_iter);
    // qDebug()<<"整体均值"<<mean_all<<"阈值"<<max_index<<"最大值时的方差"<<variance_temp[max_index]<<"小部分的均值"<<mean_temp[max_index]/p[max_index]<<"总和"<<mean_temp[255]<<"个数"<<p[255];
    auto temp=RGB2GRAY(image.clone());
    auto tmp=Threshold_Segmentation(temp,max_index);
    image_qt_result=cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);

};

void MainWindow::on_image_trans_clicked()//切换到图片变换功能的图层 按键
{
    ui->stackedWidget_2->setCurrentIndex(0);
}

void MainWindow::on_image_filter_clicked()//切换到图片滤波功能的图层 按键
{
    ui->stackedWidget_2->setCurrentIndex(1);
}

void MainWindow::on_start_clicked()//切换到图片显示开始界面的图层 按键
{
    ui->stackedWidget->setCurrentIndex(0);
}

void MainWindow::on_image_raw_clicked()//切换到图片显示原图的图层 按键
{
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow::on_image_result_clicked()//切换到图片显示结果的图层 按键
{
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_gray_clicked()//转灰度图 按键
{
    image_result=RGB2GRAY(image_raw);
    image_qt_result=cvMat2QImage(image_result,false);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_hsv_clicked()//转hsv 按键
{
    image_result=RGB2HSV(image_raw);
    image_qt_result=cvMat2QImage(image_result,false);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_horizon_clicked()//水平镜像 按键
{
    image_result=Horizontal_Mirroring(image_raw);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_vertical_clicked()//垂直镜像 按键
{
    image_result=Vertical_Mirroring(image_raw);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}


void MainWindow::on_reverse_2_clicked()//图片反转 按键
{
    image_result=Reverse(image_raw);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_default_segmentation_clicked()//默认阈值分割 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Threshold_Segmentation(tmp,127);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}
void MainWindow::on_color_segmentation_clicked()//彩色阈值分割  按键
{
    image_result=Threshold_Segmentation(image_raw,127);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}


void MainWindow::on_action1_triggered()//打开默认图片1 按键
{
    filename=":/new/prefix1/images/DSC07807.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_action2_triggered()//打开默认图片2 按键
{
    filename=":/new/prefix1/images/DSC07809.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};
void MainWindow::on_action3_triggered()//打开默认图片3 按键
{
    filename=":/new/prefix1/images/DSC08077.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_action4_triggered()//打开默认图片4 按键
{
    filename=":/new/prefix1/images/DSC08079.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_action5_triggered()//打开默认图片5 按键
{
    filename=":/new/prefix1/images/DSC08080.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_action6_triggered()//打开默认图片6 按键
{
    filename=":/new/prefix1/images/DSC08085.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_action7_triggered()//打开默认图片7 按键
{
    filename=":/new/prefix1/images/DSC08087.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_action8_triggered()//打开默认图片8 按键
{
    filename=":/new/prefix1/images/DSC08088.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_action9_triggered()//打开默认图片9 按键
{
    filename=":/new/prefix1/images/DSC08089.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_action10_triggered()//打开默认图片10 按键
{
    filename=":/new/prefix1/images/DSC08091.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_actionstart_triggered()//打开默认图片start 按键
{
    filename=":/new/prefix1/start.jpg";
    if(! ( image_qt.load(filename) ) ) //加载图像
    {
        qDebug()<<"打开图像失败";
        return;
    }
    if(check(image_qt)==false)
    {   
        ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
        ui->stackedWidget->setCurrentIndex(1);
    }
};

void MainWindow::on_mean_3_clicked()//均值滤波3 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Mean_Filtering(tmp,3);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_mean_5_clicked()//均值滤波5 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Mean_Filtering(tmp,5);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_mean_7_clicked()//均值滤波七 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Mean_Filtering(tmp,7);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_mean_12_clicked()//高斯3 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Gaussian_Filtering(tmp,3);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_mean_13_clicked()//高斯五 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Gaussian_Filtering(tmp,5);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_noise_clicked()//加噪声 按键
{
    image_raw=addSaltNoise(image_raw,3000);
    image_qt=cvMat2QImage(image_raw,true);
    ui->label_2->setPixmap(QPixmap::fromImage(image_qt));
    ui->stackedWidget->setCurrentIndex(1);
}

void MainWindow::on_mean_9_clicked()//中值滤波3 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Median_Filtering(tmp,3);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_mean_10_clicked()//中值滤波5 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Median_Filtering(tmp,5);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_mean_11_clicked()//中值滤波7 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Median_Filtering(tmp,7);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_mean_15_clicked()//X 方向sobel 按键
{
    auto tmp=RGB2GRAY(image_raw);
    cv::Mat blurred_image;
    cv::GaussianBlur(tmp, blurred_image, cv::Size(5, 5), 0.5);
    image_result=Sobel_Filtering(blurred_image,0);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_mean_16_clicked()//Y 方向sobel 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Sobel_Filtering(tmp,1);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_mean_18_clicked()//Laplace 滤波 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Laplace_Filtering(tmp);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_reverse_gray_clicked()//灰度反转 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Reverse(tmp);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_save_result_clicked()//保存结果 按键
{
    QString fileName = QFileDialog::getSaveFileName(this,
                                                  tr("保存图片"),
                                                  QDir::homePath() + "/untitled.png", // 提供默认文件名
                                                  tr("Image Files (*.png *.xpm *.jpg)"));

    if (!fileName.isEmpty()) {
        bool saved = image_qt_result.save(fileName);
        if (saved) {
            QMessageBox::information(nullptr, tr("Save Image"), tr("Image saved successfully."));
        } else {
            QMessageBox::warning(nullptr, tr("Save Image"), tr("Failed to save image."));
        }
    }
}
 
void MainWindow::on_Erosion_2_clicked()//腐蚀 按键
{
    auto temp=RGB2GRAY(image_raw);
    auto tmp=Threshold_Segmentation(temp,127);
    image_result=Erosion(tmp);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);

}

void MainWindow::on_Dilation_2_clicked()//膨胀 按键
{
    auto temp=RGB2GRAY(image_raw);
    auto tmp=Threshold_Segmentation(temp,127);
    image_result=Dilation(tmp);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_open_clicked()//开运算，先腐蚀后膨胀 按键
{
    auto temp=RGB2GRAY(image_raw);
    auto tmp=Threshold_Segmentation(temp,127);
    image_result=Erosion(tmp);
    image_result=Dilation(image_result);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_close_clicked()//闭运算，先膨胀后腐蚀 按键
{
    auto temp=RGB2GRAY(image_raw);
    auto tmp=Threshold_Segmentation(temp,127);
    image_result=Dilation(tmp);
    image_result=Erosion(image_result);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}



void MainWindow::on_reduce_laplace_clicked()//Laplace 滤波后用原图相减，做锐化 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=tmp-Laplace_Filtering(tmp);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_abs_clicked()//xy sobel后绝对值相加 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Sobel_Filtering(tmp,0)+Sobel_Filtering(tmp,1);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_histograam_clicked()//灰度直方图均衡 按键
{
    auto tmp=RGB2GRAY(image_raw);
    image_result=Histogram_Equalization(tmp);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_hist_color_clicked()//彩色直方图均衡 按键
{
    image_result=Histogram_Equalization(image_raw);
    image_qt_result=cvMat2QImage(image_result,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_horizontalSlider_valueChanged(int value)//放大缩小滑动条
{
    auto tmp=RGB2GRAY(image_raw);
    tmp=Scaling(image_raw,(float)value/100);
    // qDebug()<<tmp.rows<<tmp.cols;
    image_qt_result=cvMat2QImage(tmp,true);
    ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
    ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_dajin_clicked()
{
    dajin(image_raw);
}

void MainWindow::on_radioButton_2_clicked()
{
    Canny_Filtering(image_raw);
}

void MainWindow::on_radioButton_clicked()
{
   auto tmp=RGB2GRAY(image_raw);
   image_result=Laplace_Filtering(tmp);
   double minValue, maxValue;
   cv::Point minIdx, maxIdx;
   // 寻找最大值和最小值
   cv::minMaxLoc(image_result, &minValue, &maxValue, &minIdx, &maxIdx);
   image_result=Threshold_Segmentation(image_result,(int)(maxValue/4));
   image_qt_result=cvMat2QImage(image_result,true);
   ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
   ui->stackedWidget->setCurrentIndex(2);
}

void MainWindow::on_rotate_slider_valueChanged(int value)
{
    image_result=Rotate(image_raw,(float)value);
    image_qt_result=cvMat2QImage(image_result,true);

    if(image_qt_result.width()>750||image_qt_result.height()>520)
    {
        float reduce_scale_;
        if((float)((float)image_qt_result.width()/750.0)>=(float)((float)image_qt_result.height()/520.0))
        {
            reduce_scale_=750/(float)image_qt_result.width();
          //qDebug()<<(float)(image_qt_result.width()/750.0)<<" "<<(float)((float)image_qt_result.height()/520.0);
        }
        else
        reduce_scale_=520/(float)image_qt_result.height();
//        qDebug()<<reduce_scale_;
        image_qt_reduced=image_qt_result.scaled((int)(image_qt_result.width()*reduce_scale_-1),
                                        (int)(image_qt_result.height()*reduce_scale_-1), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->label->setPixmap(QPixmap::fromImage(image_qt_reduced));
        ui->stackedWidget->setCurrentIndex(2);
    }
    else 
    {
        ui->label->setPixmap(QPixmap::fromImage(image_qt_result));
        ui->stackedWidget->setCurrentIndex(2);
    }
}
