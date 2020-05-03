#include <iostream> 
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <map>


using namespace std;
using namespace cv;

Mat thresh;
int max_t = 255;
int t = 140;
RNG rng(12345);
Mat drawing;
vector<vector<Point> > contours;
vector<Point> points;

//acestea sunt bgr
Scalar black(30, 30, 30);
Scalar brown(0, 38, 61);
Scalar red(2, 30, 130);
Scalar orange(0, 75, 195);
Scalar yellow(30, 205, 230);//?
Scalar green(2, 85, 15);
Scalar blue(180, 30, 30);
Scalar violet(70, 27, 70);
Scalar gray(128, 128, 128);
Scalar white(220, 220, 220);
Scalar silver(169, 169, 169);
Scalar gold(70, 140, 160);
Scalar magenta(255, 0, 255);
vector<Scalar> colors = { black,brown,red,orange,yellow,green,blue,violet,gray,white,gold,silver };

bool comparator(pair<int, int> t1, pair<int, int> t2) {
    if (t2.second == t1.second) return t2.first < t1.first;
    else return t2.second < t1.second;
}

void detectContours(int, void*, int t)
{
    Mat canny_output;
    vector<Vec4i> hierarchy;
    /// Detect edges using canny
    Canny(thresh, canny_output, t, t * 2, 3);

    /// Find contours
    findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    /// Draw contours
    drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
    }

    /// Show in a window
    //namedWindow("Contours", WINDOW_NORMAL);
    //imshow("Contours", drawing);
}


vector<Mat> crops;


void detectColors(Mat rotatedImage, Mat imgOut, Point2f coord) {
        Size matSize = rotatedImage.size();
        Point pt1(0, matSize.height / 2 - 15);
        Point pt2(matSize.width, matSize.height / 2 - 15);
        Point pt3(0, matSize.height / 2 + 15);
        Point pt4(matSize.width, matSize.height / 2 + 15);
        Mat crop;
        crop = rotatedImage(Rect2i(pt1, pt4)); //am facut o noua regiune de interes (am decupat 30 de linii)
        crops.push_back(crop);
        //cout << "Size crop " << crop.rows << " " << crop.cols << endl;
        //namedWindow("newRoi", WINDOW_NORMAL);
        //imshow("newRoi", crop);

        ///////////////////////////////////////////////

        //cout << crop.rows << " " << crop.cols << endl;

        vector<int> R, G, B, SADs;
        map<int, int> histB, histG, histR;
        Vec3b  mean;
        vector<Vec3b> means;
        for (int i = 0; i < crop.cols; i++) {
            for (int j = 0; j < crop.rows; j++) {
                if ((int)crop.at<Vec3b>(j, i)[0] != (int)magenta[0] && (int)crop.at<Vec3b>(j, i)[1] != (int)magenta[1] && (int)crop.at<Vec3b>(j, i)[2] != (int)magenta[2]) {
                    B.push_back((int)crop.at<Vec3b>(j, i)[0]);
                    G.push_back((int)crop.at<Vec3b>(j, i)[1]);
                    R.push_back((int)crop.at<Vec3b>(j, i)[2]);
                }
               
            }
            
            //cout << endl;
            if (B.empty()) continue;
            sort(B.begin(), B.end());
            sort(G.begin(), G.end());
            sort(R.begin(), R.end());
            mean[0] = B[B.size() / 2];
            mean[1] = G[G.size() / 2];
            mean[2] = R[R.size() / 2];
            means.push_back(mean);
            B.clear();
            G.clear();
            R.clear();
        }//am creat o singura linie cu crop.cols pixeli care contin valoarea mediata pe fiecare coloana == means
        //cout << crop.at<Vec3b>(10,23) << " culoarea de 15  si 49" << endl;
        
        
        for (int i = 0; i < means.size(); i++) {
            histB[means[i][0]]++;
            histG[means[i][1]]++;
            histR[means[i][2]]++;
        }
        /*
        for (auto i : histB) {
            cout << i.first << " " << i.second << endl;
        }
        cout << endl;
        for (auto i : histG) {
            cout << i.first << " " << i.second << endl;
        }
        cout << endl;
        for (auto i : histR) {
            cout << i.first << " " << i.second << endl;
        }
        */
        vector< pair<int, int> > vectB(histB.begin(), histB.end());
        sort(vectB.begin(), vectB.end(), comparator);

        vector< pair<int, int> > vectG(histG.begin(), histG.end());
        sort(vectG.begin(), vectG.end(), comparator);

        vector< pair<int, int> > vectR(histR.begin(), histR.end());
        sort(vectR.begin(), vectR.end(), comparator);
        cout << endl;
        //facem histogramele ca sa vedem care e cea mai des intalnita culoare aka beige


        Vec3b reference = Vec3b(vectB[0].first, vectG[0].first, vectR[0].first);
        cout <<endl<< "culoarea de referinta etse: " << reference << endl;
        
        
        double sum = 0;
        for (int i = 0; i < means.size(); i++) {

            int SAD = abs((int)means[i][0] - (int)reference[0]) + abs((int)means[i][1] - (int)reference[1]) + abs((int)means[i][2] - (int)reference[2]);
            SADs.push_back(SAD);
            sum += SAD;
        }
        cout <<endl<< "Asta este SADs ul si anume dist intre means si reference" << endl;
        for (auto k : SADs) cout << k << " ";
        cout << endl;

        
        
        
        
        int max = -1, contor = 0;
        double average = 0.73 * sum / SADs.size();
        cout <<  endl<< "Asta este media: " << average << endl;


        vector<int> maxes, indexes, distances;
        for (int i = 0; i < SADs.size() - 1; i++) {
           
            if (SADs[i] > average) {
                cout << "1 ";
                if (SADs[i] >= max)
                    max = SADs[i];
                if (SADs[i + 1] <= average) {
                    maxes.push_back(max);
                    indexes.push_back(i); //0-64
                    max = -1;
                    if (contor != 0) {
                        distances.push_back(contor);
                        contor = 0;
                    }
                }
            }
            else {
                cout << "0 ";
                if (maxes.size() != 0) {
                    contor++;
                }
            }
        }
        cout << endl;
        
        
        
        
        
        cout <<endl<< "Aici afisam maximele locale" << endl;
        for (int i = 0; i < maxes.size(); i++) {
            cout << maxes[i] << " ";
           // cout << indexes[i] << endl;
        }


        cout <<endl<< "Aici afisam pozitiile maximelor: " << endl;
        for (auto p : indexes) cout << p << " ";
        cout << endl;


        cout <<endl<< "Aici afisam distantele intre maximele locale: " << endl;
        for (auto l : distances) {
            cout << l << " ";
        }
        cout << endl;
        
        
        vector<Scalar> myColor;
        for (int i = 0; i < indexes.size(); i++) {
            //cout << means[indexes[i]] << endl; //de la pozitiile la care am gasit maximele, mergeam in means si luam culorile
            myColor.push_back((Scalar)means[indexes[i]]);
        }
        


        cout << endl<<"Aici afisam culorile pe care le gasim la pozitiile maximelor: " << endl;
        for(auto j:myColor){
            cout << j << " ";
        }//culorile de la maxime
        cout << endl;
        
        
        
        vector <Scalar> newColors;
        vector <int> index_min, newSADs;
        for (int i = 0; i < myColor.size(); i++) {
            for (int j = 0; j < colors.size(); j++) {
                int newSAD = abs((int)myColor[i][0] - (int)colors[j][0]) + abs((int)myColor[i][1] - (int)colors[j][1]) + abs((int)myColor[i][2] - (int)colors[j][2]);
                newSADs.push_back(newSAD);
            }

            cout <<endl<< "Aici avem diferentele intre culoarea benzii " << i << " si culorile de referinta: " << endl;
            for (auto u : newSADs) cout << u << " ";
            cout << endl;


            int min = (*min_element(newSADs.begin(), newSADs.end()));
            for (int i = 0; i < newSADs.size(); i++) {
                if (newSADs[i] == min) {
                    index_min.push_back(i);
                }
            }
            newSADs.clear();
            //cout << endl;
        } //am calculat diferentele intre culorile noastre si culorile de referinta si unde diferenta e minima, inseamna ca am gasit culoarea corecta





        vector<string> name_colors;
        cout <<endl<< "aici am pozitiile la care gasesc distantele minime:" << endl;
        for (int i = 0; i < index_min.size(); i++) {
            newColors.push_back(colors[index_min[i]]);
            cout << index_min[i] << " ";
            switch (index_min[i]) {
            case 0: name_colors.push_back("black"); break;
            case 1: name_colors.push_back("brown"); break;
            case 2: name_colors.push_back("red"); break;
            case 3: name_colors.push_back("orange"); break;
            case 4: name_colors.push_back("yellow"); break;
            case 5: name_colors.push_back("green"); break;
            case 6: name_colors.push_back("blue"); break;
            case 7: name_colors.push_back("violet"); break;
            case 8: name_colors.push_back("gray"); break;
            case 9: name_colors.push_back("white"); break;
            case 10: name_colors.push_back("gold"); break;
            case 11: name_colors.push_back("silver"); break;
            }
        }




        vector<double> tolerance = { 0,1,2,0,0,0.5,0.25,0.1,0.05,0,5,10,20 };
        vector<string> reverse_name_colors(name_colors.rbegin(), name_colors.rend());
        string nameR;
       
        
        switch (indexes.size()) {
            //case 3:




            case 4:
                if (distances[0] >= distances[2]) {//toleranta este prima
                    long int value = (index_min[3] * 10 + index_min[2]) * pow(10, index_min[1]);
                    nameR = to_string(value) + "ohm " + to_string((int)(tolerance[index_min[0]])) + "%";
                    for (auto c : reverse_name_colors) cout << c << " ";
                    cout << value << "ohm " << tolerance[index_min[0]] << "%" << endl;
                }
                else {
                    long int value = (index_min[0] * 10 + index_min[1]) * pow(10, index_min[2]);
                    nameR = to_string(value) + "ohm " + to_string((int)(tolerance[index_min[3]])) + "%";
                    for (auto c : name_colors) cout << c << " ";
                    cout << value << "ohm " << tolerance[index_min[3]] << "%" << endl;
                }
                break;

            case 5:
                if (distances[0] >= distances[3]) {//toleranta este prima
                    long int value = (index_min[4] * 100 + index_min[3] * 10 + index_min[2]) * pow(10, index_min[1]);
                    nameR = to_string(value) + "ohm " + to_string((int)(tolerance[index_min[0]])) + "%";
                    for (auto c : reverse_name_colors) cout << c << " ";
                    cout << value << "ohm " << tolerance[index_min[0]] << "%" << endl;
                }
                else {
                    long int value = (index_min[0] * 100 + index_min[1] * 10 + index_min[2]) * pow(10, index_min[3]);
                    nameR = to_string(value) + "ohm " + to_string((int)(tolerance[index_min[4]])) + "%";
                    for (auto c : name_colors) cout << c << " ";
                    cout << value << "ohm " << tolerance[index_min[4]] << "%" << endl;
                }
                break;
        }

        putText(imgOut, nameR, coord, FONT_HERSHEY_DUPLEX, 1.5, Scalar(0, 0, 0), 2);
        namedWindow("valoare Rezistenta", WINDOW_NORMAL);
        imshow("valoare Rezistenta", imgOut);
        name_colors.clear();
    
}

int show_histogram(std::string const& name, cv::Mat1b const& image)
{
    // Set histogram bins count
    int bins = 256;
    int histSize[] = { bins };
    // Set ranges for histogram bins
    float lranges[] = { 0, 256 };
    const float* ranges[] = { lranges };
    // create matrix for histogram
    cv::Mat hist;
    int channels[] = { 0 };
    // create matrix for histogram visualization
    int const hist_height = 256;
    cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    double max_val = 0;
    Point max_loc;
    minMaxLoc(hist, 0, &max_val, 0, &max_loc);
    // visualize each bin
    for (int b = 0; b < bins; b++) {
        float const binVal = hist.at<float>(b);
        int   const height = cvRound(binVal * hist_height / max_val);
        line(hist_image, Point(b, hist_height - height), Point(b, hist_height), Scalar::all(255));
    }
    //imshow(name, hist_image);
    return max_loc.y;
}

int main(int argc, char argv[])
{
    Mat imgIn = imread("pic2.jpg");
    if (imgIn.empty())
    {
        //Print error message
        cout << "Error occured when loading the image" << endl;
        return -1;
    }

    //Sum the colour values in each channel
    Scalar sumImg = sum(imgIn);
    //normalise by the number of pixels in the image to obtain an extimate for the illuminant
    Scalar illum = sumImg / (imgIn.rows * imgIn.cols);
    // Split the image into different channels
    vector<Mat> rgbChannels(3);
    split(imgIn, rgbChannels);
    //Assign the three colour channels to CV::Mat variables for processing
    Mat redImg = rgbChannels[2];
    Mat greenImg = rgbChannels[1];
    Mat blueImg = rgbChannels[0];
    //calculate scale factor for normalisation you can use 255 instead
    double scale = (illum(0) + illum(1) + illum(2)) / 3;
    //correct for illuminant (white balancing)
    redImg = redImg * scale / illum(2);
    greenImg = greenImg * scale / illum(1);
    blueImg = blueImg * scale / illum(0);
    //Assign the processed channels back into vector to use in the cv::merge() function
    rgbChannels[0] = blueImg;
    rgbChannels[1] = greenImg;
    rgbChannels[2] = redImg;
    Mat imgOut; //to hold the output image
    //Merge the processed colour channels
    merge(rgbChannels, imgOut);
    //namedWindow("clean", WINDOW_NORMAL);
    //imshow("clean", imgOut);
    



    Mat binaryImage;
    cvtColor(imgOut, binaryImage, COLOR_BGR2GRAY);
    //namedWindow("binaryImage", WINDOW_NORMAL);
    //imshow("binaryImage", binaryImage);
    //imwrite("fara normlaizare.jpg", imgOut);
    



    Mat gauss;
    GaussianBlur(binaryImage, gauss, Size(7, 7), 0);
    //namedWindow("gauss", WINDOW_NORMAL);
    //imshow("gauss", gauss);
    //imwrite("file.png", gauss);
    int max_loc = show_histogram("hist", gauss);
    cout << max_loc << endl;
    gauss.convertTo(gauss,CV_32FC1,1.0/255.0);
    gauss *=(255.0 / max_loc);
    gauss.convertTo(gauss,CV_8UC1,255/1);
    normalize(gauss, gauss, 0, 255, NORM_MINMAX);
    int max2_loc = show_histogram("newHist", gauss);
    cout << max2_loc << endl;
    //namedWindow("gauss after norm", WINDOW_NORMAL);
    //imshow("gauss after norm", gauss);

   

    threshold(gauss, thresh, t, 255, THRESH_BINARY_INV);
    //namedWindow("threshImage", WINDOW_NORMAL);
    //imshow("threshImage", thresh);




    Mat kernel = getStructuringElement(MORPH_RECT, Point(20, 20));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);
    morphologyEx(thresh, thresh, MORPH_OPEN, kernel);
    //namedWindow("morphologyEx", WINDOW_NORMAL);
    //imshow("morphologyEx", thresh);

    detectContours(0, 0, t);

    normalize(imgOut, imgOut, 0, 512, NORM_MINMAX);
    Mat normalizedImg = imgOut.clone();
    //namedWindow("normalize", WINDOW_NORMAL);
    //imshow("normalize", normalizedImg); //ca sa vedem imaginea initiala
    
    
    
    RotatedRect box;
    Point2f vtx[4]; //varfurile dreptunghiului
    vector<Point2f> vtxs;
    for (int i = 0; i < contours.size(); i += 2) {
        //cout << contour << " ";
        box = minAreaRect(contours[i]);
        // draw box
        box.points(vtx);
        for (int i = 0; i < 4; i++) {
            line(imgOut, vtx[i], vtx[(i + 1) % 4], Scalar(0, 0, 255), 3, LINE_AA);
        }
        vtxs.push_back(vtx[2]);
    }
    //namedWindow("minAreaRect", WINDOW_NORMAL);
    //imshow("minAreaRect", imgOut);
    
    

    
    vector<Mat> images;
    Mat rotatedImage;
    //calcul unghiului cu care trebuie sa rotim rezistorul
    //cout << vtx[0] << " " << vtx[1] << " " << vtx[2] << " " << vtx[3] << endl;
    double AB = norm(Mat(vtx[0]), Mat(vtx[1]));
    double AD = norm(Mat(vtx[0]), Mat(vtx[3]));

    
    
    for (int i = 0; i < contours.size(); i += 2) {
        //cout << contours.size() << " ";
        Rect boundingBox = boundingRect(contours[i]);
        RotatedRect rotatedRect = minAreaRect(contours[i]);
        Mat roi = normalizedImg(boundingBox);
        
        double angle;
        if (AB > AD) angle = rotatedRect.angle+90;
        else angle = rotatedRect.angle;
        if (angle == 0) angle = rotatedRect.angle;
        //cout << AB << " " << AD << " " <<angle<<endl;
        
        Mat rotationMatrix = getRotationMatrix2D(Point2f(roi.cols / 2, roi.rows / 2), angle, 0.7);
        //cout << endl<< rotatedRect.angle << endl;
        normalize(roi, roi, 0, 275, NORM_MINMAX);
        warpAffine(roi, rotatedImage, rotationMatrix, Size(roi.cols, roi.rows) , 1, 0, magenta);
        //namedWindow("rotatedImage", WINDOW_NORMAL);
        //imshow("rotatedImage", rotatedImage);
        images.push_back(rotatedImage);
    }




    //pentru cand avem mai multe rezistoare intr-o imagine
    String im = "image";
    String ro = "roi";
    int j = 1;
    for (int i = 0; i < images.size(); i++) {
        string name = im + to_string(j);
        string roii = ro + to_string(j);
        //cout << name << endl;
        namedWindow(name, WINDOW_NORMAL);
        imshow(name, images[i]);
        j++;
        if (images[i].rows > 30) {
            detectColors(images[i], imgOut, vtxs[i]);
            namedWindow(roii, WINDOW_NORMAL);
            imshow(roii, crops[i]);
        }
    }


    cv::waitKey();
    return 0;
}


