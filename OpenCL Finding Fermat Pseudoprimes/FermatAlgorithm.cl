__kernel void fermat(__global int* input, __global int* result);
__kernel void getModofExp(__local int* res, int base, int exp, int mod);

__kernel void fermat(__global int* input, __global int* result) {
	size_t id = get_global_id(0);
	int n = input[id];
	__local int m;
	for(int i=2;i<n;i++) {
		getModofExp(&m, i, n-1, n);
		if(m != 1) {
			result[n] = 0;
			return;
		}
	}
	result[n] = 1;
}

__kernel void getModofExp(__local int* res, int base, int exp, int mod) {
	int c = 1;
	int exp_prime = 0;
	while(exp_prime < exp) {
	    exp_prime++;
		c = (base*c) % mod;
	}
	*res = c;
}