#include "Fern.h"
#include <filesystem>
#include "cnpy.h"
void Fern::load_model(std::string pth)
{

	//cnpy::NpyArray arr = cnpy::npy_load(pth);
	//Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 2>> index_pair(arr.data<double>());
	//
	using Ftype = float;
	using EigenXfR = Eigen::Matrix<Ftype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using EigenXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using EigenXiR = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	using EigenXuiR = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	std::string directory_name = "fern_regressor_" + m_name;
	std::filesystem::path save_path(pth);
	save_path /= directory_name;

	cnpy::NpyArray bin_param_data = cnpy::npy_load((save_path/"bin_param.npy").string());
	cnpy::NpyArray Fern_threshold_data = cnpy::npy_load((save_path/"Fern_threshold.npy").string());
	cnpy::NpyArray sel_pix_index_data = cnpy::npy_load((save_path/"selected_pixel_index.npy").string());

	int param_num = bin_param_data.shape[0];
	int row = bin_param_data.shape[1];
	int col = bin_param_data.shape[2];
	for (int i = 0; i < param_num; i++) {
		Eigen::Map<EigenXfR> param_i(bin_param_data.data<Ftype>() + i*row*col, row, col);
		m_bin_params.push_back(param_i.cast<double>());
	}
	//std::cout << Eigen::Map<EigenXdR>(Fern_threshold_data.data<double>(), Fern_threshold_data.shape[0], Fern_threshold_data.shape[1]) << std::endl << std::endl;
	m_Fern_threshold = Eigen::Map<EigenXdR>(Fern_threshold_data.data<double>(), Fern_threshold_data.shape[0], Fern_threshold_data.shape[1]);
	//std::cout << Eigen::Map<EigenXiR>(sel_pix_index_data.data<int>(), sel_pix_index_data.shape[0], sel_pix_index_data.shape[1]).cast<int>() << std::endl;
	m_selected_pixel_index = Eigen::Map<EigenXuiR>(sel_pix_index_data.data<unsigned int>(), sel_pix_index_data.shape[0], sel_pix_index_data.shape[1]).cast<int>();
}

void Fern::predict(const Eigen::VectorXd& intensity_vector, Eigen::MatrixXd& cur_pose) {
	int index = 0;
	for (int i = 0; i < m_F; i++) {

		int pidx1 = m_selected_pixel_index(i, 0);
		int pidx2 = m_selected_pixel_index(i, 1);
		double val = intensity_vector(pidx1) - intensity_vector(pidx2);
		//std::cout << m_selected_pixel_index << std::endl;
		//std::cout << val << std::endl;
		//std::cout << i << std::endl;
		//std::cout << m_Fern_threshold << std::endl;
		if (val >= m_Fern_threshold(i, 0) ) {
			index +=  pow(2, i);
		}

	}
	cur_pose += m_bin_params[index];
}
