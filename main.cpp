#include<iostream>
#include<vector>
#include<stdexcept>
#include<cmath>

struct Matrix {
	std::vector<std::vector<float>> data;

	Matrix(int r, int c) : data(r, std::vector<float>(c, 0.0f)) {}

	Matrix(std::initializer_list<std::vector<float>> init): data(init) {}



	int rows() const { return data.size(); }
	int cols() const { return data.empty() ? 0 : data[0].size(); }

	void print() const {
		for(int i = 0;i < rows(); ++i){
			for(int j = 0; j < cols(); ++j){
				std::cout<< data[i][j]<<" ";
			}
			std::cout<<"\n";
		}

		std::cout<< "-------------------------------\n";		
	}
};

Matrix matmul(const Matrix& A, const Matrix& B){
	if(A.cols() != B.rows()) {
		throw std::invalid_argument("Shape mismatch, rows and cols must be same");
	}
	// output shape
	Matrix C(A.rows(), B.cols());

	// standard O(n^3) dot product code
	
	for(int i=0;i<A.rows();++i){
		for(int j=0;j<B.cols();++j){
			for(int k=0;k<A.cols();++k){
				C.data[i][j] += A.data[i][k] * B.data[k][j];
			}
		}
	}
	return C;	
}


Matrix transpose(const Matrix& A){
	Matrix C(A.cols(), A.rows());
	for(int i=0;i<A.rows();++i){
		for(int j=0;j<A.cols();++j){
			C.data[j][i] = A.data[i][j];
		}
	}
	return C;
}


Matrix scale(Matrix A, float scalar){
	for(int i=0;i<A.rows();++i){
		for(int j=0;j<A.cols();++j){
			A.data[i][j] *= scalar;
		}
	}
	return A;
}

Matrix apply_mask(Matrix A, const Matrix& mask) {// masking is a mathematical technique used during the attention mechanism to hide specific wordsor tokens from the model.
	for(int i=0;i<A.rows();++i){
		for(int j=0;j<A.cols();++j){
			if(mask.data[i][j] == 0.0f){ // assume mask uses 0 to indicate hidden token
				A.data[i][j] = -1e9f; // set to big ass -ve number
			}
		}
	}
	return A;
}
//Softmax: Converts raw scores into a probability distribution (weights)
Matrix softmax(Matrix A) {
    for (int i = 0; i < A.rows(); ++i) {
        // Find the max value in the row for numerical stability
        // (Prevents std::exp from blowing up and returning NaN)
        float max_val = A.data[i][0];
        for (int j = 1; j < A.cols(); ++j) {
            if (A.data[i][j] > max_val) {
                max_val = A.data[i][j];
            }
        }
        
        float sum_exp = 0.0f;
        // Compute exponentials and sum them
        for (int j = 0; j < A.cols(); ++j) {
            A.data[i][j] = std::exp(A.data[i][j] - max_val);
            sum_exp += A.data[i][j];
        }
        
        // Divide by the sum to get probabilities that sum to 1.0
        for (int j = 0; j < A.cols(); ++j) {
            A.data[i][j] /= sum_exp;
        }
    }
    return A;
}


Matrix scaled_dot_product_attention(const Matrix& Q, const Matrix& K, const Matrix& V, const Matrix& mask = {}) {
	// 1. Q * K^T
	Matrix K_T = transpose(K);
	Matrix scores = matmul(Q,K_T);
	// Scale by 1 / sqrt(d_k)
	float d_k = static_cast<float>(K.cols());
	scores = scale(scores,1.0f / std::sqrt(d_k));
	// apply optional mask (used in decoder to prevent looking into future)
	if(!mask.data.empty()) {
		scores = apply_mask(scores,mask); // sets msked positions to -infy
	}

	Matrix weights = softmax(scores);// softmax to get attention weights

	return matmul(weights,V); // matmul with V
				  //
	
}

int main() {
    // Query [2x3]
    Matrix Q = {
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 0.0}
    };

    // Key [2x3]
    Matrix K = {
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 0.0}
    };

    // Value [2x4] 
    // (We use 4 dimensions here to prove Attention can change the embedding size)
    Matrix V = {
        {9.0, 9.0, 9.0, 9.0}, // Token 1's actual value
        {5.0, 5.0, 5.0, 5.0}  // Token 2's actual value
    };

    std::cout << "Running Full Scaled Dot-Product Attention...\n";
    
    // Run the full pipeline
    Matrix final_output = scaled_dot_product_attention(Q, K, V);

    std::cout << "Final Contextualized Output Matrix [2x4]:\n";
    final_output.print();

    return 0;
}