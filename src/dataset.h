#pragma once
#include "string"
#include <Eigen/Core>

#include <vector>

struct Dataset {


	
	Dataset() {};


	void load(const std::string& pth);

	std::vector<Eigen::MatrixXd> m_Ss_list;
	std::vector<Eigen::MatrixXd> m_S_original_list;
	std::vector<int> m_S_init_list;
	std::vector<Eigen::MatrixXd> m_Rt_inv_list;
	std::vector<int> m_SRt_index_list;




};