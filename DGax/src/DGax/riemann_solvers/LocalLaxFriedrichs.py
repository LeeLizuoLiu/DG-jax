import jax.numpy as jnp

def local_lax_friedrichs_matlab(uM, uP, normals, equations):
        
    lambdaM = equations.max_wave_speed(uM) 
    lambdaP = equations.max_wave_speed(uP) 
    lambda_val = jnp.maximum(lambdaM, lambdaP) 
        
    FluxP = equations.flux(uP) 
    FluxM = equations.flux(uM)

    diffusion_term =  lambda_val*(uM - uP)
    face_integral = jnp.einsum('pd, pvd -> pv', normals, FluxP + FluxM) + diffusion_term 
    return face_integral