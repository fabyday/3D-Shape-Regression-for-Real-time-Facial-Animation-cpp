#include <string>
#include <Eigen/Core>

class Fern {

public : 
	Fern(int F, int beta, int P, std::string name) : m_F(F), m_beta(beta), m_P(P), m_name(name){};
	void load_model(std::string pth);
	void predict(const Eigen::VectorXd& intensity_vector, Eigen::MatrixXd& cur_pose);



private:
	std::string m_name;
	int m_beta;
	int m_F;
	int m_P;

	Eigen::MatrixXd m_Fern_threshold;
	Eigen::MatrixXi m_selected_pixel_index;
	std::vector<Eigen::MatrixXd> m_bin_params;

};