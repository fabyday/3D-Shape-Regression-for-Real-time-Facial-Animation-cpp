
#pragma once 
#include <eigen/Core>
#include <string>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <filesystem>
#include "cnpy.h"

inline bool compareNat(const std::string& a, const std::string& b)
{
	if (a.empty())
		return true;
	if (b.empty())
		return false;
	if (std::isdigit(a[0]) && !std::isdigit(b[0]))
		return true;
	if (!std::isdigit(a[0]) && std::isdigit(b[0]))
		return false;
	if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
	{
		if (std::toupper(a[0]) == std::toupper(b[0]))
			return compareNat(a.substr(1), b.substr(1));
		return (std::toupper(a[0]) < std::toupper(b[0]));
	}

	// Both strings begin with digit --> parse both numbers
	std::istringstream issa(a);
	std::istringstream issb(b);
	int ia, ib;
	issa >> ia;
	issb >> ib;
	if (ia != ib)
		return ia < ib;

	// Numbers are the same --> remove numbers and recurse
	std::string anew, bnew;
	std::getline(issa, anew);
	std::getline(issb, bnew);
	return (compareNat(anew, bnew));
}


struct Blendshapes{



	mutable Eigen::MatrixXd neutral_pose;
	//Eigen::MatrixXd pose;
	std::vector<Eigen::MatrixXd> pose;
	Eigen::MatrixXd flaten_pose;
	Eigen::MatrixXi F;

	void compile() {
		int row_size = pose[0].size();
		int col_size = pose.size();
		Eigen::Map<Eigen::MatrixXd> flat_neutral_pose(neutral_pose.data(), neutral_pose.size(), 1);
		flaten_pose.conservativeResize(row_size, col_size);
		for (int i = 0; i < col_size; i++) {
			//flaten_pose.block(i,0, row_size, 1) = flat_neutral_pose - Eigen::Map<Eigen::MatrixXd>(pose[i].data(), row_size, 1);
			//flaten_pose.block(0, i, row_size, 1) = flat_neutral_pose - Eigen::Map<Eigen::MatrixXd>(pose[i].data(), row_size, 1);
			flaten_pose.col(i) = Eigen::Map<Eigen::MatrixXd>(pose[i].data(), row_size, 1) - flat_neutral_pose;
		}
	};
int get_blendshape_weight_size() const {
	return flaten_pose.cols();
	}
void blend( const Eigen::Ref<const Eigen::VectorXd> w, Eigen::MatrixXd& out) const {
	Eigen::Map<Eigen::MatrixXd> flat_neutral_pose(neutral_pose.data(), neutral_pose.size(), 1);

	Eigen::MatrixXd flat_mesh = flat_neutral_pose + flaten_pose * w;
	Eigen::Map<Eigen::MatrixXd> tmp_map(flat_mesh.data(), neutral_pose.rows(), neutral_pose.cols());
	out = tmp_map;
};
void blend(Eigen::Ref<const Eigen::VectorXd> w, Eigen::MatrixXd& out, const std::vector<int>& lmk) const {
	out.conservativeResize(lmk.size(), 3);
	
	Eigen::Map<Eigen::MatrixXd> flat_neutral_pose( neutral_pose.data(), neutral_pose.size(), 1);
	//std::cout << w << std::endl;


	//Eigen::MatrixXd q(lmk.size(),3);
	//for (int i = 0; i < lmk.size(); i++) {
	//	q.row(i) = neutral_pose.row(lmk[i]);
	//}	
	//std::cout << "neu" << q.block(0, 0, 4, 3) << std::endl;

	//

	//for (int i = 0; i < flaten_pose.cols(); i++) {

	//Eigen::MatrixXd t = flaten_pose.col(i);
	//Eigen::Map<Eigen::MatrixXd> dt(t.data(), int(flaten_pose.col(0).size() / 3), 3);
	//Eigen::MatrixXd q(lmk.size(),3);
	//for (int i = 0; i < lmk.size(); i++) {
	//	q.row(i) = dt.row(lmk[i]);
	//}	
	//std::cout <<i << "th- mat" << std::endl;
	//std::cout << q.block(0,0,4,3) << std::endl;
	//}


	Eigen::MatrixXd mesh = neutral_pose + (flaten_pose * w);

	Eigen::Map<Eigen::MatrixXd> tmp_map(mesh.data(), neutral_pose.rows(), neutral_pose.cols());
	for (int i = 0; i < lmk.size(); i++) {
		out.row(i) = tmp_map.row(lmk[i]);
	}
	//std::cout<< out << std::endl;
};
}
;

inline void load_blendshapes(const std::string& root_pth, struct Blendshapes& bb) {
	using namespace std;

	filesystem::path root = root_pth;
	cout << root << endl;
	if (!filesystem::is_directory(root)) {
		cerr << "this path is not dir." << endl;
		return;
	}
	std::vector<string> pth_list;
	filesystem::path neutral_pth;

	for ( auto pth : filesystem::directory_iterator(root)) {
		
		if (pth.path().extension() == ".obj") {
			
			if (pth.path().string().find("identity") != std::string::npos) {
				neutral_pth = pth.path();
			}
			else {
				pth_list.push_back(pth.path().string());

			}
		}

	}
	
	sort(pth_list.begin(), pth_list.end(), compareNat);

	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::cout << "neutral apth " << neutral_pth.string() << std::endl;
	
	igl::read_triangle_mesh(neutral_pth.string(), V, F);
	bb.neutral_pose = V;
	bb.F = F;

	std::vector<Eigen::MatrixXd> exprs;
	for (int i = 0; i < pth_list.size(); i++) {
		std::cout << "blendshapes : " << pth_list[i] << "was loaded." << std::endl;
		igl::read_triangle_mesh(pth_list[i], V, F);
		exprs.emplace_back(V);
	}
	bb.pose = exprs;
	bb.compile();
}

#include <Eigen/Geometry>

struct TransformRecord{
	std::vector<Eigen::MatrixXd> m_prev_translations;
	std::vector<Eigen::Quaterniond> m_prev_rotations;
	int window_size;
	int current_index;
	int stacked_size;

	TransformRecord(int window_size = 5) :window_size(5), stacked_size(0), current_index(-1) {
		m_prev_translations.resize(window_size);
		m_prev_rotations.resize(window_size);
	};
	void update_prev_transform(Eigen::MatrixXd& translation, Eigen::Quaterniond& rot) {
		m_prev_translations[current_index] = translation;
		m_prev_rotations[current_index] = rot;
	}
	void push_prev_transform(Eigen::MatrixXd& translation, Eigen::Quaterniond& rot) {
		
		current_index = (current_index+1)%window_size;
		m_prev_translations[current_index] = translation;
		m_prev_rotations[current_index] = rot;
		stacked_size++;
	}

	void get_info(int& window_size, int& stack_size, int& cur_idx) {
		window_size = this->window_size;
		stack_size = this->stacked_size;
		cur_idx = this->current_index;
	}

};	
#include "gaussian_mixture_model.h"
struct FicalTrackingParams {
	Eigen::MatrixXd m_neutral;
	Eigen::MatrixXd m_pose;
	
	TransformRecord prev_transforms;
	BLendshapes_GMM* m_gmm;
	std::vector<Eigen::MatrixXd> prev_coeffs;
	int window_size = 5;
	int updated_frame = 0;
	int coeff_index;
public:
	FicalTrackingParams(Blendshapes& blendshapes, std::vector<int>& lmk, BLendshapes_GMM* gmm)
		:prev_transforms(gmm != nullptr ? gmm->get_winsize() : 5), m_gmm(gmm)
	{

		coeff_index = -1;
		this->window_size = window_size;
		m_pose.conservativeResize(lmk.size() * 3, blendshapes.get_blendshape_weight_size());
		m_neutral.conservativeResize(lmk.size(), 3);
		prev_coeffs.resize(window_size, Eigen::MatrixXd::Ones(blendshapes.get_blendshape_weight_size(), 1)*0.0);
		Eigen::Map<Eigen::MatrixXd> flat_neutral_pose(blendshapes.neutral_pose.data(), blendshapes.neutral_pose.size(), 1);

		int row_size = blendshapes.neutral_pose.rows();

		//Eigen::VectorXd tw = Eigen::VectorXd::Ones(blendshapes.get_blendshape_weight_size());

		
		int lmk_size = lmk.size();
		//std::cout <<"size " << lmk_size << std::endl;
		
		for (int i = 0; i < lmk_size; i++) {
			//std::cout << "======" << std::endl;
			//std::cout <<std::setw(3) << i << "/" << lmk_size << "," << std::setw(5 ) <<row_size * 0 + lmk[i] <<"/"<< row_size*3 << std::endl;
			//std::cout <<std::setw(3) << i + 1 * lmk_size << "/" << lmk_size << "," << std::setw(5) << row_size * 1 + lmk[i] << "/" << row_size * 3 << std::endl;
			//std::cout <<std::setw(3) << i + 2 * lmk_size << "/" << lmk_size << "," << std::setw(5) << row_size * 2 + lmk[i] << "/" << row_size * 3 << std::endl;
			m_pose.row(i) = blendshapes.flaten_pose.row(row_size * 0 + lmk[i]);
			m_pose.row(i + 1*lmk_size) = blendshapes.flaten_pose.row(row_size * 1 + lmk[i]);
			m_pose.row(i + 2*lmk_size) = blendshapes.flaten_pose.row(row_size * 2 + lmk[i]);
		}


		for (int i = 0; i < lmk.size(); i++) {
			m_neutral.row(i) = blendshapes.neutral_pose.row(lmk[i]);
		}
	}
	int get_blendshape_weight_size() const {
		return m_pose.cols();
	}

	void update_prev_blendshapes(Eigen::MatrixXd& coeff) {
		coeff_index = (window_size + 1) % window_size;
		prev_coeffs[coeff_index] = coeff;
	}
	void update_prev_blendshapes(Eigen::VectorXd& coeff) {
		coeff_index = (window_size + 1) % window_size;
		prev_coeffs[coeff_index] = coeff;
		++updated_frame;
	}
	void get_prev_coeff(Eigen::VectorXd& coeff) {
		coeff_index = (window_size - 1) % window_size;
		coeff = prev_coeffs[coeff_index];
	}
	int get_window(Eigen::VectorXd& res) {
		if (updated_frame < window_size)
			return 0;
		int w_size = get_blendshape_weight_size();
		res.conservativeResize(window_size * w_size);
		auto index = coeff_index;
		for (int i = 0; i < window_size; i++) {
			res.segment(i * w_size, w_size) = prev_coeffs[(index + i) % window_size];
		}
		return 1;
	}

	void blend(const Eigen::VectorXd& w, Eigen::MatrixXd& out) {

		Eigen::Map<Eigen::MatrixXd> flat_neutral_pose(m_neutral.data(), m_neutral.size(), 1);
		Eigen::MatrixXd res = (flat_neutral_pose + (m_pose * w));
		out = Eigen::Map<Eigen::MatrixXd>(res.data(), m_neutral.rows(), 3);

	};

};





inline void load_blendshape_weight_animation(const std::string& root_pth, Eigen::MatrixXd& out) 
{


		std::filesystem::path root = root_pth;

		std::filesystem::path meta_path = "";
		std::vector<std::string> pths;
 		for (auto const& dir_entry : std::filesystem::directory_iterator{ root }) {
			std::string ext = dir_entry.path().extension().string();
			if (ext == ".npy")
				pths.emplace_back(dir_entry.path().string());
		}



		// regressor init
		out;
		std::vector<cnpy::NpyArray> list;
		int mat_rows = 0;
		int w_size = 0;
		for (int i = 0; i < pths.size(); i++) {

			cnpy::NpyArray arr = cnpy::npy_load(pths[i]);
			//cnpy::npz_t arrs= cnpy::npz_load(pths[i]);//, "ws");
			//arrs;
			//cnpy::NpyArray arr;
			list.push_back(arr);                   
			mat_rows += arr.shape[0];
			w_size = arr.shape[1];
			//int w_num = arr.shape[1];
		}
		out.conservativeResize(mat_rows, w_size);
		int cur_index = 0;
		for (int i = 0; i < list.size(); i++) {
			out.block(cur_index, 0, list[i].shape[0], list[i].shape[1]) = Eigen::Map < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(list[i].data<double>(), list[i].shape[0], list[i].shape[1]);
			cur_index = list[i].shape[0];
		}
}