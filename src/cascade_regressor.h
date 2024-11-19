#pragma once
#include "dataset.h"
#include <string>


#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include "weak_regressor.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include "blendshapes.h"
class CascadeRegressor {




public:

	CascadeRegressor() : m_T(0), m_K(0), m_F(0), m_P(0), m_beta(0), m_lmk_size(0),  m_Q(), m_S_list(){

	};

	void load(const std::string& pth);
	void add_dataset(Dataset* d) {
		m_dataset = d;

		m_meanshape.conservativeResizeLike(m_dataset->m_Ss_list[0]);
		m_meanshape.setZero();
		for (int i = 0; i<m_dataset->m_Ss_list.size(); i++)
			m_meanshape += m_dataset->m_Ss_list[i];
		m_meanshape /= m_dataset->m_Ss_list.size();

	};
	void add_blendshapes(struct Blendshapes* blend) {
		m_blendshapes = blend;
	};

	void load_dlib_detector(const std::string& dlib_pth) {

		m_detector = dlib::get_frontal_face_detector();
		dlib::deserialize(dlib_pth) >> m_shape_predictor;


	};

	void set_autogen_lmk_index(std::vector<int>& lmk) {
		m_auotogen_lmk_index = lmk;
	}

	void set_68landmark_index(std::vector<int>& lmk) {
		m_lmk_index = lmk;
	}
	bool predict(const cv::Mat& img, const Eigen::MatrixXd* prev_shapes, Eigen::MatrixXd& result_shape, int init_num = 15);

	bool get_landmark_pos(const cv::Mat& img, Eigen::MatrixXd& out);
	Eigen::Matrix3d get_Q() { return m_Q; };

private:
	std::vector<WeakRegressor> m_weak_regs;

	int m_T, m_K, m_F, m_P, m_beta, m_lmk_size;
	Eigen::Matrix3d m_Q;
	std::vector<Eigen::MatrixXd> m_S_list;
	Dataset* m_dataset;


	Eigen::MatrixXd m_meanshape;

	Blendshapes* m_blendshapes;
	std::vector<int> m_lmk_index;
	std::vector<int> m_auotogen_lmk_index; 
	dlib::frontal_face_detector m_detector;
	dlib::shape_predictor m_shape_predictor;
};