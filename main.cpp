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

// hella unnecessary bullshit code but modern implementations are such anyways:

// build a slicer that imitates numpy.split() takes a matrix, slices out a vector out of it (vertical chunk mind you!)

Matrix slice_cols(const Matrix& A, int start_col, int end_col){
	Matrix C(A.rows(), end_col-start_col);

	for(int i=0;i<A.rows();++i){
		for(int j= start_col; j<end_col;++j){
			C.data[i][j-start_col] = A.data[i][j];
		}
	}
	return C;
}

// now we need a function that takes two matrices and glues them together side-by-side.

Matrix concat_cols(const Matrix& A, const Matrix& B){
	
	if(A.rows() != B.rows()){
		throw std::invalid_argument("Row mismatch, stitching operation not possible for matrices A and B");
	}
	Matrix C(A.rows(), A.cols()+B.cols());

	for(int i=0;i<A.rows();++i){
		for(int j=0;j<A.cols();++j){
			C.data[i][j] = A.data[i][j];
		}
		for(int k=0;k<B.cols();++k){
			C.data[i][k+A.cols()] = B.data[i][k];
		}
	}
	return C;

}


Matrix multi_head_attention(const Matrix& Q, const Matrix& K, const Matrix& V, int num_heads) {
    // Check if the embedding dimension can be evenly split
    if (Q.cols() % num_heads != 0) {
        throw std::invalid_argument("Embedding dimension must be divisible by num_heads");
    }

    int head_dim = Q.cols() / num_heads;
    
    // We will build the final output by iteratively stitching heads together
    Matrix final_output(Q.rows(), 0); 

    for (int h = 0; h < num_heads; ++h) {
        int start_col = h * head_dim;
        int end_col = (h + 1) * head_dim;

        // 1. Slice the matrices for this specific head
        Matrix Q_head = slice_cols(Q, start_col, end_col);
        Matrix K_head = slice_cols(K, start_col, end_col);
        Matrix V_head = slice_cols(V, start_col, end_col);

        // 2. Run the attention engine on this smaller chunk
        Matrix head_result = scaled_dot_product_attention(Q_head, K_head, V_head);

        // 3. Stitch it into the master output
        if (h == 0) {
            final_output = head_result; // The first head becomes the base matrix
        } else {
            final_output = concat_cols(final_output, head_result); // Append to the right
        }
    }

    return final_output;
}

// ReLU Activation
Matrix relu(Matrix A) {
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            if (A.data[i][j] < 0.0f) {
                A.data[i][j] = 0.0f; // Zero out negative values
            }
        }
    }
    return A;
}

//The Feed-Forward Block
Matrix feed_forward(const Matrix& X, const Matrix& W1, const Matrix& W2) {
    // Step 1: Project to the hidden layer
    Matrix hidden = matmul(X, W1);
    
    // Apply non-linearity
    hidden = relu(hidden);
    
    //Project back to the original embedding dimension
    return matmul(hidden, W2);
}


Matrix layer_norm(Matrix A) {
    // A tiny number to prevent division by zero if variance is exactly 0
    float epsilon = 1e-5f; 

    for (int i = 0; i < A.rows(); ++i) {
        
        // Calculate the Mean (Average) of the row
        float sum = 0.0f;
        for (int j = 0; j < A.cols(); ++j) {
            sum += A.data[i][j];
        }
        float mean = sum / A.cols();

        // Calculate the Variance (Average of squared differences from the mean)
        float variance_sum = 0.0f;
        for (int j = 0; j < A.cols(); ++j) {
            float diff = A.data[i][j] - mean;
            variance_sum += diff * diff;
        }
        float variance = variance_sum / A.cols();

        // Normalize the row
        for (int j = 0; j < A.cols(); ++j) {
            A.data[i][j] = (A.data[i][j] - mean) / std::sqrt(variance + epsilon);
        }
    }
    
    // Returning the modified matrix
    return A; 
}

Matrix add(const Matrix& A, const Matrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Shape mismatch: Cannot add matrices of different sizes.");
    }
    Matrix C(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            C.data[i][j] = A.data[i][j] + B.data[i][j];
        }
    }
    return C;
}

int main() {
    std::cout << "--- Initializing Bare-Metal Transformer Engine ---\n\n";

    // The Input Sequence (X)
    // 2 tokens, embedding dimension of 4.
    Matrix X = {
        {1.0f, 0.5f, -0.2f, 0.8f}, // Token 1
        {0.2f, -1.0f, 0.5f, 0.1f}  // Token 2
    };

    std::cout << "Input Embeddings [2x4]:\n";
    X.print();

    // ==========================================
    // BLOCK 1: Multi-Head Attention + Add & Norm
    // ==========================================
    
    // In self-attention, X acts as Query, Key, and Value
    int num_heads = 2; // Split the 4 dimensions into 2 heads (2 dims each)
    Matrix mha_output = multi_head_attention(X, X, X, num_heads);
    
    // Residual Connection: Add original input to attention output
    Matrix residual_1 = add(X, mha_output);
    
    // Layer Normalization
    Matrix norm_1 = layer_norm(residual_1);

    std::cout << "After Multi-Head Attention & LayerNorm 1 [2x4]:\n";
    norm_1.print();

    // ==========================================
    // BLOCK 2: Feed-Forward Network + Add & Norm
    // ==========================================
    
    // FFN Weights
    // W1 expands dimension from 4 to 8
    Matrix W1 = {
        {0.1f, 0.2f, -0.1f, 0.5f, 0.1f, 0.0f, -0.2f, 0.3f},
        {-0.2f, 0.1f, 0.4f, -0.1f, 0.2f, -0.1f, 0.5f, 0.0f},
        {0.3f, -0.3f, 0.1f, 0.2f, -0.4f, 0.2f, 0.1f, 0.1f},
        {0.0f, 0.4f, -0.2f, 0.1f, 0.3f, 0.1f, -0.1f, 0.2f}
    };
    
    // W2 compresses dimension back from 8 to 4
    Matrix W2 = {
        {0.2f, -0.1f, 0.3f, 0.1f},
        {0.1f, 0.2f, -0.2f, 0.0f},
        {-0.3f, 0.1f, 0.1f, 0.2f},
        {0.2f, -0.2f, 0.0f, 0.3f},
        {0.1f, 0.1f, 0.2f, -0.1f},
        {-0.1f, 0.3f, -0.1f, 0.1f},
        {0.4f, 0.0f, -0.2f, 0.2f},
        {0.0f, 0.2f, 0.1f, -0.3f}
    };

    // Run through the FFN
    Matrix ffn_output = feed_forward(norm_1, W1, W2);
    
    // Residual Connection: Add norm_1 to FFN output
    Matrix residual_2 = add(norm_1, ffn_output);
    
    // Final Layer Normalization
    Matrix final_encoder_output = layer_norm(residual_2);

    std::cout << "Final Encoder Output Matrix [2x4]:\n";
    final_encoder_output.print();
    std::cout << "--- Forward Pass Complete ---\n";

    return 0;
}