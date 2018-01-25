
class RefineLayer:
	def __init__(self, inputs):
		self.variable = {}
		self.variable["inputs"] = inputs
		self.start_numeric = False

	# refine layer ---------------------------------------------------------------------------------
	def refine(self, inputs):
		in_sampling3D, in_sampling2D, in_objPts, in_imgPts, in_hyp, in_objIdx, in_shuffleIdx, in_cmat, in_distcoeffs = inputs
		self.inlierMaps = np.zeros((in_hyp.shape[0], in_sampling3D.shape[0]))
		
		def _refine(sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeffs):
			refHyps = np.zeros(hyps.shape, dtype=np.float64)	
			samplingCopy = copy.deepcopy(sampling3D)
			'''
			for i in range(objPts.shape[0]):
				samplingCopy[objIdx[i]] = copy.deepcopy(objPts[i])
			'''
			for h in range(refHyps.shape[0]):
				done0, rot0, tran0 = cv2.solvePnP(objPts[h], imgPts[h], cmat, distcoeffs)
				newHyp = np.append(rot0, tran0)
				diffmaps = getDiffMap(newHyp, samplingCopy, sampling2D,cmat, distcoeffs)
				for i in range(cfg.REFTIMES):
					inlier3D = []
					inlier2D = []
					for idx in shuffleIdx[i]:
						if diffmaps[idx] < cfg.INLIERTHRESHOLD2D:
							inlier3D.append(samplingCopy[idx])
							inlier2D.append(sampling2D[idx])
							inlierMaps[h][idx] = 1
						if len(inlier3D) > cfg.INLIERCOUNT:
							break
					if len(inlier3D) < 3:
						continue
					refineObj = np.array(inlier3D)
					refinePt = np.array(inlier2D)
					done, rot, tran = cv2.solvePnP(refineObj, refinePt, cmat, distcoeffs, False, cv2.SOLVEPNP_ITERATIVE if refineObj.shape[0] >= 4 else cv2.SOLVEPNP_P3P)
					if containNan(rot) or containNan(tran):
						break
					refHyps[h] = np.append(rot, tran)
					diffmaps = getDiffMap(refHyps[h], samplingCopy, sampling2D, cmat, distcoeffs)
				for idx in objIdx[h]:
					self.inlierMaps[h][idx] = 0
			print(inlierMaps)
			return refHyps

		def _refine_single(sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeffs):
			refHyps = np.zeros(hyps.shape, dtype=np.float64)
			for h in range(refHyps.shape[0]):
				diffmaps = getDiffMap(hyps[h], sampling3D, sampling2D, cmat, distcoeffs)
				for i in range(cfg.REFTIMES):
					inlier3D = []
					inlier2D = []
					for idx in shuffleIdx[i]:
						if diffmaps[idx] < cfg.INLIERTHRESHOLD2D:
							inlier3D.append(sampling3D[idx])
							inlier2D.append(sampling2D[idx])
						if len(inlier3D) > cfg.INLIERCOUNT:
							break
					if len(inlier3D) < 3:
						continue
					refineObj = np.array(inlier3D)
					refinePt = np.array(inlier2D)
					done, rot, tran = cv2.solvePnP(refineObj, refinePt, cmat, distcoeffs, False, cv2.SOLVEPNP_ITERATIVE if refineObj.shape[0] >= 4 else cv2.SOLVEPNP_P3P)
					if containNan(rot) or containNan(tran):
						break
					refHyps[h] = np.append(rot, tran)
					#print("i:", i, refHyps[h])
					diffmaps = getDiffMap(refHyps[h], sampling3D, sampling2D, cmat, distcoeffs)
			return refHyps

		# @param grad, dLoss wrt hyp, hx6, actually the same shape as hyps
		def _refine_grad(sampling3D, sampling2D, objPts, imgPts, hyps, objIdx, shuffleIdx, cmat, distcoeffs, grad):
			# print(grad)
			# dRefine wrt the picked up points, which were used to generate pose before
			# numeric method
			eps = 1
			jacobean_obj = np.zeros((objPts.shape[0], objPts.shape[1], objPts.shape[2]), np.float64)
			jacobean_sample = np.zeros_like(sampling3D, np.float64)	
			return jacobean_obj, jacobean_sample

		def _refine_grad_op(op, grad):
			[sampling3D, sampling2D, objPts, imgPts, thyp, objIdx, shuffleIdx, cmat, distcoeffs] = op.inputs

			dObj, dSample = tf.py_func(_refine_grad, [sampling3D, sampling2D, objPts, imgPts, thyp, objIdx, shuffleIdx, cmat, distcoeffs, grad], [tf.float64, tf.float64])		
			return [dSample, None, dObj, None, None, None, None, None, None]
		
		grad_name = "RefineGrad_" + str(uuid.uuid4())
		tf.RegisterGradient(grad_name)(_refine_grad_op)
		g = tf.get_default_graph()
		with g.gradient_override_map({"PyFunc": grad_name}):
			output = tf.py_func(_refine, inputs, tf.float64)
		return output

reflayer = RefineLayer([tf_sample3D, tf_sample2D, tf_objPts, tf_imgPts, tf_hyp, tf_objIdx, tf_shuffleIdx, tf_cmat, tf_D])
out = reflayer.refine([tf_sample3D, tf_sample2D, tf_objPts, tf_imgPts, tf_hyp, tf_objIdx, tf_shuffleIdx, tf_cmat, tf_D])
