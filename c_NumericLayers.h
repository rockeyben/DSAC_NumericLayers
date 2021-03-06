void c_mySolvePnP(double * input_array1, double * input_array2, double * input_array3, 
	double * output_array, int methodFlag,
	int size11, int size12, int size13,
	int size21, int size22, int size23,
	int size31, int size32,
	int size41, int size42);

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
	int refSteps, 
	int inlier_count);

void c_refine_single(double * ref_hyp,
	double * sampling3D,
	double * sampling2D,
	double * hyp,
	int * shuffleIdx,
	double * camMat,
	int n,
	int hyp_num,
	int ref_steps,
	int inlier_count);

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
	int skip);