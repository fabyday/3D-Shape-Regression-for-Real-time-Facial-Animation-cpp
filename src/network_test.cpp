#include "network.h"
int main() {



	networkCtx nctx;
	CProcessInfo pctx;
 	create_child_viewer(pctx);
	create_client_socket(nctx);
	Eigen::VectorXd t(57);
	t.setZero();
	int id = 0;
	while (true) {
		t(id) += 0.1;
		if (t(id) >= 1.0) {
			t(id++) = 1;
			id %= 57;
		}
		Sleep(100);
		send_blendshapes(t, nctx);
	}



	WaitForSingleObject(pctx.pi.hProcess, INFINITE);
}