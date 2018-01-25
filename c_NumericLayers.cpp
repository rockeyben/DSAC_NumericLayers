#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <iomanip>
#include "c_NumericLayers.h"
#include <iostream>

#define CNN_OBJ_MAXINPUT 100.0
#define INLIERTHRESHOLD2D 10
// input_array1: objPts nx4x3
// input_array2: imgPts nx4x2
// input_array3: camMat 3x3
// output_array: hyps nx6
void c_mySolvePnP(double * input_array1, double * input_array2, double * input_array3, 
	double * output_array, int methodFlag,
	int size11, int size12, int size13,
	int size21, int size22, int size23,
	int size31, int size32,
	int size41, int size42);

bool containsNaNs(const cv::Mat& m)
{
   return cv::sum(cv::Mat(m != m))[0] > 0;
}

bool safeSolvePnP(
    std::vector<cv::Point3d> objPts1,
    const std::vector<cv::Point2d>& imgPts1,
    const cv::Mat& camMat,
    const cv::Mat& distCoeffs,
    cv::Mat& rot,
    cv::Mat& trans,
    bool extrinsicGuess,
    int methodFlag)
{
    if(!cv::solvePnP(objPts1, imgPts1, camMat, distCoeffs, rot, trans, extrinsicGuess,methodFlag))
    {
        rot = cv::Mat_<double>::zeros(3, 1);
        trans = cv::Mat_<double>::zeros(3, 1);
        return false;
    }
    return true;
}


std::vector<float> getDiffMap(
  	const std::vector<double> hyp,
	const std::vector<cv::Point3d>& sampling3D,
	const std::vector<cv::Point2d>& sampling2D,
  	const cv::Mat& camMat)
{
	std::vector<float> diffMap(sampling3D.size());

	cv::Mat hyp_rot(3, 1, CV_64F);
	cv::Mat hyp_trans(3, 1, CV_64F);
	for(int i = 0; i < 3; i++)
		hyp_rot.at<double>(i, 0) = hyp[i];
	for(int i = 0; i < 3; i++)
		hyp_trans.at<double>(i, 0) = hyp[i+3];

	std::vector<cv::Point2d> projections;
	cv::projectPoints(sampling3D, hyp_rot, hyp_trans, camMat, cv::Mat(), projections);


	// measure reprojection errors
	for(int i = 0; i < projections.size(); i++)
	{
		cv::Point2d curPt = sampling2D[i] - projections[i];
		float l = std::min(cv::norm(curPt), CNN_OBJ_MAXINPUT);
		diffMap[i] = l;
	}

	return diffMap;
}



void c_mySolvePnP(double * input_array1, double * input_array2, double * input_array3, 
	double * output_array, int methodFlag,
	int size11, int size12, int size13,
	int size21, int size22, int size23,
	int size31, int size32,
	int size41, int size42)
{
	
	std::vector<std::vector<cv::Point3d> > objPts;
	std::vector<std::vector<cv::Point2d> > imgPts;
	std::vector<cv::Mat> rot_list;
	std::vector<cv::Mat> trans_list;
	cv::Mat_<double> camMat = cv::Mat_<double>::zeros(3,3);
	
	for(int i = 0; i < size31; i++)
	for(int j = 0; j < size32; j++)
		camMat(i, j) = input_array3[i * size32 + j];


	objPts.resize(size11);
	imgPts.resize(size21);
	rot_list.resize(size41);
	trans_list.resize(size41);

	#pragma omp parallel for
	for(int i = 0; i < size11; i++)
	for(int j = 0; j < size12; j++)
	{
		int b1 = i*size12*size13 + j*size13;
		int b2 = i*size22*size23 + j*size23;
		objPts[i].push_back(cv::Point3d(input_array1[b1], input_array1[b1+1], input_array1[b1+2]));
		imgPts[i].push_back(cv::Point2d(input_array2[b2], input_array2[b2+1]));
	}
	
	#pragma omp parallel for
	for(int h = 0; h < size11; h++)
	{
		cv::solvePnP(objPts[h], imgPts[h], camMat, cv::Mat(), rot_list[h], trans_list[h], false, CV_ITERATIVE);
	}

	#pragma omp parallel for
	for(int i = 0; i < size41; i++)
	{
		for(int j = 0; j < size42 / 2; j++)
			output_array[i*size42 + j] = rot_list[i].at<double>(j, 0);
		for(int j = size42/2; j < size42; j++)
			output_array[i*size42 + j] = trans_list[i].at<double>(j-size42/2, 0);
	}
	
	return ;
}


std::vector<double> refine_single(
	int inlier_count,
	int ref_steps,
	const std::vector<std::vector<int> >& shuffleIdx,
	const std::vector<cv::Point3d>& sampling3D,
	const std::vector<cv::Point2d>& sampling2D,
	const std::vector<double> hyp,
	const cv::Mat& camMat)
{
	cv::Mat hyp_rot(3, 1, CV_64F);
	cv::Mat hyp_trans(3, 1, CV_64F);
	cv::Mat hypUpdate_rot(3, 1, CV_64F);
	cv::Mat hypUpdate_trans(3, 1, CV_64F);
	for(int i = 0; i < 3; i++)
		hyp_rot.at<double>(i, 0) = hyp[i];
	for(int i = 0; i < 3; i++)
		hyp_trans.at<double>(i, 0) = hyp[i+3];

	std::vector<double> new_hyp(hyp);

	std::vector<float> diffMap = getDiffMap(hyp, sampling3D, sampling2D, camMat);

	for(int rStep = 0; rStep < ref_steps; rStep++)
	{
		 // collect 2D-3D correspondences
        std::vector<cv::Point2d> localImgPts;
        std::vector<cv::Point3d> localObjPts;

        for(int i = 0; i < shuffleIdx[rStep].size(); i++)
        {
        	int idx = shuffleIdx[rStep][i];

        	// inlier check
        	if(diffMap[idx] < INLIERTHRESHOLD2D)
        	{
        		localObjPts.push_back(sampling3D[idx]);
        		localImgPts.push_back(sampling2D[idx]);
        	}

        	if(localImgPts.size() >= inlier_count)
        		break;
        }

        if(localImgPts.size() < 3)
        	continue;

        // recalculate pose

        hypUpdate_rot = hyp_rot.clone();
        hypUpdate_trans = hyp_trans.clone();

        if(!safeSolvePnP(localObjPts, localImgPts, camMat, cv::Mat(), hypUpdate_rot, hypUpdate_trans, false, (localImgPts.size() > 4) ? CV_ITERATIVE : CV_P3P))
        	break; // abort if PnP fails
        if(containsNaNs(hypUpdate_rot) || containsNaNs(hypUpdate_trans))
            break; // abort if PnP fails

        for(int i = 0; i < 3; i++)
			new_hyp[i] = hypUpdate_rot.at<double>(i, 0);
		for(int i = 0; i < 3; i++)
			new_hyp[i+3] = hypUpdate_trans.at<double>(i, 0);

        diffMap = getDiffMap(new_hyp, sampling3D, sampling2D, camMat);
	}

	return new_hyp;
}

// sampling3D: (n, 3)
// sampling2D: (n, 2)
// objPts:     (h, 4, 3)
// imgPts:     (h, 4, 2)
// hyps:       (h, 6)
// objIdx:     (h, 4)
// shuffleIdx: (refinesteps, n)
// inlier_map  (h, n)
// camMat:     (3, 3)
void c_refine(double * ref_hyps, 
	double * sampling3D, 
	double * sampling2D, 
	double * objPts, 
	double * imgPts, 
	double * hyps,
	int * inlier_map, 
	int * objIdx, 
	int * shuffleIdx, 
	double * camMat,
	int n,
	int hyp_num, 
	int init_num, 
	int ref_steps,
	int inlier_count)
{
	std::vector<cv::Point3d> sampling3D_vec;
	std::vector<cv::Point2d> sampling2D_vec;
	std::vector<std::vector<double> > hyp_vec;
	std::vector<std::vector<int> > objIdx_vec;
	std::vector<std::vector<int> > shuffleIdx_vec;
	std::vector<std::vector<cv::Point3d> > objPts_vec;
	std::vector<std::vector<cv::Point2d> > imgPts_vec;
	cv::Mat_<double> m_camMat = cv::Mat_<double>::zeros(3,3);
	
	for(int i = 0; i < 3; i++)
	for(int j = 0; j < 3; j++)
		m_camMat(i, j) = camMat[i * 3 + j];

	for(int i = 0; i < n; i++)
	{
		sampling3D_vec.push_back(cv::Point3d(sampling3D[i*3], sampling3D[i*3+1], sampling3D[i*3+2]));
		sampling2D_vec.push_back(cv::Point2d(sampling2D[i*2], sampling2D[i*2+1]));
	}

	for(int i = 0; i < hyp_num; i++)
	{
		std::vector<double> tmp_hyp(hyps+i*6, hyps+i*6+6);
		std::vector<int> tmp_idx(objIdx+i*4, objIdx+i*4+4);
		hyp_vec.push_back(tmp_hyp);
		objIdx_vec.push_back(tmp_idx);

		std::vector<cv::Point3d> obj;
		std::vector<cv::Point2d> img;

		for(int j = 0; j < init_num; j++)
		{
			obj.push_back(cv::Point3d(objPts[i*init_num*3+j*3], objPts[i*init_num*3+j*3+1], objPts[i*init_num*3+j*3+2]));
			img.push_back(cv::Point2d(imgPts[i*init_num*2+j*2], imgPts[i*init_num*2+j*2+1]));
		}

		objPts_vec.push_back(obj);
		imgPts_vec.push_back(img);
	}

	for(int i = 0; i < ref_steps; i++)
	{
		std::vector<int> tmp_shuffle(shuffleIdx+i*n, shuffleIdx+i*n+n);
		shuffleIdx_vec.push_back(tmp_shuffle);
	}

	std::vector<std::vector<double> > ref_hyps_vec;
	ref_hyps_vec.resize(hyp_num);

	#pragma omp parallel for
	for(int h = 0; h < hyp_num; h++)
	{
		std::vector<double> hyp = hyp_vec[h];
		cv::Mat hyp_rot(3, 1, CV_64F);
		cv::Mat hyp_trans(3, 1, CV_64F);
		cv::Mat hypUpdate_rot(3, 1, CV_64F);
		cv::Mat hypUpdate_trans(3, 1, CV_64F);
		for(int i = 0; i < 3; i++)
			hyp_rot.at<double>(i, 0) = hyp[i];
		for(int i = 0; i < 3; i++)
			hyp_trans.at<double>(i, 0) = hyp[i+3];

		std::vector<double> new_hyp(hyp);
		std::vector<float> diffMap = getDiffMap(hyp, sampling3D_vec, sampling2D_vec, m_camMat);

		for(int rStep = 0; rStep < ref_steps; rStep++)
		{
			 // collect 2D-3D correspondences
	        std::vector<cv::Point2d> localImgPts;
	        std::vector<cv::Point3d> localObjPts;

	        for(int i = 0; i < shuffleIdx_vec[rStep].size(); i++)
	        {
	        	int idx = shuffleIdx_vec[rStep][i];

	        	// inlier check
	        	if(diffMap[idx] < INLIERTHRESHOLD2D)
	        	{
	        		localObjPts.push_back(sampling3D_vec[idx]);
	        		localImgPts.push_back(sampling2D_vec[idx]);
	        		inlier_map[h*n+idx] += 1;
	        	}

	        	if(localImgPts.size() >= inlier_count)
	        		break;
	        }

	        if(localImgPts.size() < 3)
	        	continue;

	        // recalculate pose

	        hypUpdate_rot = hyp_rot.clone();
	        hypUpdate_trans = hyp_trans.clone();

	        if(!safeSolvePnP(localObjPts, localImgPts, m_camMat, cv::Mat(), hypUpdate_rot, hypUpdate_trans, false, (localImgPts.size() > 4) ? CV_ITERATIVE : CV_P3P))
	        	break; // abort if PnP fails
	        if(containsNaNs(hypUpdate_rot) || containsNaNs(hypUpdate_trans))
	            break; // abort if PnP fails

	        for(int i = 0; i < 3; i++)
				new_hyp[i] = hypUpdate_rot.at<double>(i, 0);
			for(int i = 0; i < 3; i++)
				new_hyp[i+3] = hypUpdate_trans.at<double>(i, 0);

	        diffMap = getDiffMap(new_hyp, sampling3D_vec, sampling2D_vec, m_camMat);
		}

		for(int i = 0; i < init_num; i++)
		{
			int idx = objIdx_vec[h][i];
			inlier_map[h*n+idx] = 0;
		}

		ref_hyps_vec[h] = new_hyp;
	}

	for(int i = 0; i < hyp_num; i++)
	for(int j = 0; j < 6; j++)
		ref_hyps[i*6+j] = ref_hyps_vec[i][j];

	return ;
}

// ref_hyp     (6,)
// sampling3D: (n, 3)
// sampling2D: (n, 2)
// hyp:        (6,)
// shuffleIdx: (refinesteps, n)
// camMat:     (3, 3)
void c_refine_single(double * ref_hyp,
	double * sampling3D,
	double * sampling2D,
	double * hyp,
	int * shuffleIdx,
	double * camMat,
	int n,
	int hyp_num,
	int ref_steps,
	int inlier_count)
{
	std::vector<cv::Point3d> sampling3D_vec;
	std::vector<cv::Point2d> sampling2D_vec;
	std::vector<double> hyp_vec(hyp, hyp+6);
	std::vector<std::vector<int> > shuffleIdx_vec;
	cv::Mat_<double> m_camMat = cv::Mat_<double>::zeros(3,3);
	
	for(int i = 0; i < 3; i++)
	for(int j = 0; j < 3; j++)
		m_camMat(i, j) = camMat[i * 3 + j];

	for(int i = 0; i < n; i++)
	{
		sampling3D_vec.push_back(cv::Point3d(sampling3D[i*3], sampling3D[i*3+1], sampling3D[i*3+2]));
		sampling2D_vec.push_back(cv::Point2d(sampling2D[i*2], sampling2D[i*2+1]));
	}

	for(int i = 0; i < ref_steps; i++)
	{
		std::vector<int> tmp_shuffle(shuffleIdx+i*n, shuffleIdx+i*n+n);
		shuffleIdx_vec.push_back(tmp_shuffle);
	}

	std::vector<double> new_hyp;
	new_hyp = refine_single(
		inlier_count, 
		ref_steps,
		shuffleIdx_vec, 
		sampling3D_vec, 
		sampling2D_vec, 
		hyp_vec,
		m_camMat);

	for(int i = 0; i < 6; i++)
		ref_hyp[i] = new_hyp[i];

	return ;
}

// jacobean_obj    (h, 4, 3)
// jacobean_sample (n, 3)
// grad            (h, 6)
// inlier_map      (h, n)
void c_dRefine(double * jacobean_obj,
	double * jacobean_sample, 
	double * sampling3D, 
	double * sampling2D, 
	double * objPts, 
	double * imgPts,
	int * inlier_map,
	int * objIdx, 
	int * shuffleIdx, 
	double * camMat,
	double * grad,
	int n,
	int hyp_num, 
	int init_num, 
	int ref_steps, 
	int inlier_count,
	double eps,
	int skip)
{
	std::vector<cv::Point3d> sampling3D_vec;
	std::vector<cv::Point2d> sampling2D_vec;
	std::vector<std::vector<double> > hyp_vec;
	std::vector<std::vector<int> > objIdx_vec;
	std::vector<std::vector<int> > shuffleIdx_vec;
	std::vector<std::vector<cv::Point3d> > objPts_vec;
	std::vector<std::vector<cv::Point2d> > imgPts_vec;
	cv::Mat_<double> m_camMat = cv::Mat_<double>::zeros(3,3);
	
	for(int i = 0; i < 3; i++)
	for(int j = 0; j < 3; j++)
		m_camMat(i, j) = camMat[i * 3 + j];

	for(int i = 0; i < n; i++)
	{
		sampling3D_vec.push_back(cv::Point3d(sampling3D[i*3], sampling3D[i*3+1], sampling3D[i*3+2]));
		sampling2D_vec.push_back(cv::Point2d(sampling2D[i*2], sampling2D[i*2+1]));
	}

	for(int i = 0; i < hyp_num; i++)
	{
		std::vector<int> tmp_idx(objIdx+i*4, objIdx+i*4+4);
		objIdx_vec.push_back(tmp_idx);

		std::vector<cv::Point3d> obj;
		std::vector<cv::Point2d> img;

		for(int j = 0; j < init_num; j++)
		{
			obj.push_back(cv::Point3d(objPts[i*init_num*3+j*3], objPts[i*init_num*3+j*3+1], objPts[i*init_num*3+j*3+2]));
			img.push_back(cv::Point2d(imgPts[i*init_num*2+j*2], imgPts[i*init_num*2+j*2+1]));
		}

		objPts_vec.push_back(obj);
		imgPts_vec.push_back(img);
	}


	for(int i = 0; i < ref_steps; i++)
	{
		std::vector<int> tmp_shuffle(shuffleIdx+i*n, shuffleIdx+i*n+n);
		shuffleIdx_vec.push_back(tmp_shuffle);
	}


	// derivative wrt initial 4 points
	/*
	#pragma omp parallel for
	for(int h = 0; h < hyp_num; h++)
	{
		for(int i = 0; i < init_num; i++)
		for(int j = 0; j < 3; j++)
		{
			std::vector<double> new_hyp1(6);
			std::vector<double> new_hyp2(6);
			std::vector<double> hyp1(6);
			std::vector<double> hyp2(6);

			cv::Mat hyp_rot(3, 1, CV_64F);
			cv::Mat hyp_trans(3, 1, CV_64F);

			switch(j)
			{
				case 0:	objPts_vec[h][i].x += eps;sampling3D_vec[objIdx_vec[h][i]].x += eps;break;
				case 1: objPts_vec[h][i].y += eps;sampling3D_vec[objIdx_vec[h][i]].y += eps;break;
				case 2: objPts_vec[h][i].z += eps;sampling3D_vec[objIdx_vec[h][i]].z += eps;break;
				default:break;
			}

			if(!safeSolvePnP(objPts_vec[h], imgPts_vec[h], m_camMat, cv::Mat(), hyp_rot, hyp_trans, false, CV_ITERATIVE))
				continue;

			for(int k = 0; k < 3; k++) hyp1[k] = hyp_rot.at<double>(k, 0);
			for(int k = 0; k < 3; k++) hyp1[k+3] = hyp_trans.at<double>(k, 0);

			new_hyp1 = refine_single(
				inlier_count, 
				ref_steps,
				shuffleIdx_vec, 
				sampling3D_vec, 
				sampling2D_vec, 
				hyp1,
				m_camMat);


			switch(j)
			{
				case 0:	objPts_vec[h][i].x -= 2 * eps;sampling3D_vec[objIdx_vec[h][i]].x -= 2 * eps;break;
				case 1: objPts_vec[h][i].y -= 2 * eps;sampling3D_vec[objIdx_vec[h][i]].y -= 2 * eps;break;
				case 2: objPts_vec[h][i].z -= 2 * eps;sampling3D_vec[objIdx_vec[h][i]].z -= 2 * eps;break;
				default:break;
			}

			if(!safeSolvePnP(objPts_vec[h], imgPts_vec[h], m_camMat, cv::Mat(), hyp_rot, hyp_trans, false, CV_ITERATIVE))
				continue;

			for(int k = 0; k < 3; k++) hyp2[k] = hyp_rot.at<double>(k, 0);
			for(int k = 0; k < 3; k++) hyp2[k+3] = hyp_trans.at<double>(k, 0);

			new_hyp2 = refine_single(
				inlier_count, 
				ref_steps,
				shuffleIdx_vec, 
				sampling3D_vec, 
				sampling2D_vec, 
				hyp2,
				m_camMat);

			switch(j)
			{
				case 0:	objPts_vec[h][i].x += eps;sampling3D_vec[objIdx_vec[h][i]].x += eps;break;
				case 1: objPts_vec[h][i].y += eps;sampling3D_vec[objIdx_vec[h][i]].y += eps;break;
				case 2: objPts_vec[h][i].z += eps;sampling3D_vec[objIdx_vec[h][i]].z += eps;break;
				default:break;
			}

			for(int k = 0; k < 6; k++)
			{
				jacobean_obj[h*init_num*3+i*3+j] += ((new_hyp1[k] - new_hyp2[k]) / (2 * eps) * grad[h*6+k]);
			}
		}
	}

	*/

	// derivatice wrt other points
	#pragma omp parallel for
	for(int h = 0; h < hyp_num; h++)
	{
		int inCount = 0;
		for(int i = 0; i < n; i++)
		{
			if(inlier_map[h*n+i] == 0)
				continue;

			inCount ++;
			if(inCount % skip == 0)
				continue;

			for(int j = 0; j < 3; j++)
			{

				std::vector<double> new_hyp1(6);
				std::vector<double> new_hyp2(6);
				std::vector<double> hyp1(6);
				std::vector<double> hyp2(6);

				cv::Mat hyp_rot(3, 1, CV_64F);
				cv::Mat hyp_trans(3, 1, CV_64F);

				switch(j)
				{
					case 0:	sampling3D_vec[i].x += eps;break;
					case 1: sampling3D_vec[i].y += eps;break;
					case 2: sampling3D_vec[i].z += eps;break;
					default:break;
				}


				if(!safeSolvePnP(objPts_vec[h], imgPts_vec[h], m_camMat, cv::Mat(), hyp_rot, hyp_trans, false, CV_ITERATIVE))
					continue;

				for(int k = 0; k < 3; k++) hyp1[k] = hyp_rot.at<double>(k, 0);
				for(int k = 0; k < 3; k++) hyp1[k+3] = hyp_trans.at<double>(k, 0);

				new_hyp1 = refine_single(
					inlier_count, 
					ref_steps,
					shuffleIdx_vec, 
					sampling3D_vec, 
					sampling2D_vec, 
					hyp1,
					m_camMat);

				switch(j)
				{
					case 0:	sampling3D_vec[i].x -= 2 * eps;break;
					case 1: sampling3D_vec[i].y -= 2 * eps;break;
					case 2: sampling3D_vec[i].z -= 2 * eps;break;
					default:break;
				}

				if(!safeSolvePnP(objPts_vec[h], imgPts_vec[h], m_camMat, cv::Mat(), hyp_rot, hyp_trans, false, CV_ITERATIVE))
					continue;

				for(int k = 0; k < 3; k++) hyp2[k] = hyp_rot.at<double>(k, 0);
				for(int k = 0; k < 3; k++) hyp2[k+3] = hyp_trans.at<double>(k, 0);

				new_hyp2 = refine_single(
					inlier_count, 
					ref_steps,
					shuffleIdx_vec, 
					sampling3D_vec, 
					sampling2D_vec, 
					hyp2,
					m_camMat);

				switch(j)
				{
					case 0:	sampling3D_vec[i].x += eps;break;
					case 1: sampling3D_vec[i].y += eps;break;
					case 2: sampling3D_vec[i].z += eps;break;
					default:break;
				}

				for(int k = 0; k < 6; k++)
				{
					jacobean_sample[i*3+j] += ((new_hyp1[k] - new_hyp2[k]) / (2 * eps) * grad[h*6+k]);
				}

			}
		}
	}
	


	return ;
}