//! # CasADi Rust interface
//!
//! This is a Rust interface to CasADi C functions.
//!
//! This is a `no-std` library (however, mind that the CasADi-generated code
//! requires `libm` to call math functions such as `sqrt`, `sin`, etc...)
//!
//! ---
//!
//! Auto-generated header file
//! This file is part of OptimizationEngine
//! (see https://alphaville.github.io/optimization-engine/)
//!
//! Generated at: {{timestamp_created}}
//!

// #![no_std]

/// Number of static parameters (this also includes penalty constraints)
const NUM_STATIC_PARAMETERS: usize = {{ problem.dim_parameters() or 0 }};

/// Number of decision variables
const NUM_DECISION_VARIABLES: usize = {{ problem.dim_decision_variables() }};

/// Number of ALM-type constraints (dimension of F1, i.e., n1)
const NUM_CONSTRAINTS_TYPE_ALM: usize = {{ problem.dim_constraints_aug_lagrangian() or 0 }};

/// Number of penalty constraints (dimension of F2, i.e., n2)
const NUM_CONSTRAINTS_TYPE_PENALTY: usize = {{ problem.dim_constraints_penalty() or 0 }};

use libc::{c_double, c_int};  // might need to include: c_longlong, c_void

/// C interface (Function API exactly as provided by CasADi)
extern "C" {
    fn cost_function_{{ meta.optimizer_name }}(
        arg: *const *const c_double,
        casadi_results: *mut *mut c_double) -> c_int;
    fn grad_cost_function_{{ meta.optimizer_name }}(
        arg: *const *const c_double,
        casadi_results: *mut *mut c_double)
        -> c_int;
    fn mapping_f1_function_{{ meta.optimizer_name }}(
        arg: *const *const c_double,
        casadi_results: *mut *mut c_double,
    ) -> c_int;
    fn mapping_f2_function_{{ meta.optimizer_name }}(
        arg: *const *const c_double,
        casadi_results: *mut *mut c_double,
    ) -> c_int;
    fn rectangle_lower_function_{{ meta.optimizer_name }}(
        arg: *const *const c_double,
        casadi_results: *mut *mut c_double,
    ) -> c_int;
    fn rectangle_upper_function_{{ meta.optimizer_name }}(
        arg: *const *const c_double,
        casadi_results: *mut *mut c_double,
    ) -> c_int;
} // END of extern C


// -----------------------------------------------------------
//  *MAIN* API Functions in Rust
// -----------------------------------------------------------


///
/// Consume the cost function psi(u, xi, p) written in C
///
/// # Example
/// ```ignore
/// fn tst_call_casadi_cost() {
///     let u = [1.0, 2.0, 3.0, -5.0, 6.0];
///     let p = [1.0, -1.0];
///     let xi = [100.0, 0.0, 1.5., 3.0];
///     let mut cost_value = 0.0;
///     icasadi::cost(&u, &xi, &p, &mut cost_value);
/// }
/// ```
///
/// # Panics
/// This method panics if the following conditions are not satisfied
///
/// - `u.len() == NUM_DECISION_VARIABLES`
/// - `static_params.len() == NUM_STATIC_PARAMETERS`
///
pub fn cost(u: &[f64], xi: &[f64], static_params: &[f64], cost_value: &mut f64) -> i32 {
    assert_eq!(u.len(), NUM_DECISION_VARIABLES, "wrong length of `u`");
    assert_eq!(
        static_params.len(),
        NUM_STATIC_PARAMETERS,
        "wrong length of `p`"
    );

    let arguments = &[u.as_ptr(), xi.as_ptr(), static_params.as_ptr()];
    let cost = &mut [cost_value as *mut c_double];

    unsafe {
        cost_function_{{ meta.optimizer_name }}(
            arguments.as_ptr(),
            cost.as_mut_ptr(),
        ) as i32
    }
}

///
/// Consume the Jacobian function written in C
///
/// # Example
/// ```ignore
/// fn tst_call_casadi_cost() {
///     let u = [1.0, 2.0, 3.0, -5.0, 6.0];
///     let p = [1.0, -1.0];
///     let xi = [100.0, 0.0, 1.5., 3.0];
///     let mut jac = [0.0; 10];
///     icasadi::grad(&u, &xi, &p, &mut jac);
/// }
/// ```
///
/// # Panics
/// This method panics if the following conditions are not satisfied
///
/// - `u.len() == icasadi::num_decision_variables()`
/// - `static_params.len() == icasadi::num_static_parameters()`
/// - `cost_jacobian.len() == icasadi::num_decision_variables()`
///
pub fn grad(u: &[f64], xi: &[f64], static_params: &[f64], cost_jacobian: &mut [f64]) -> i32 {
    assert_eq!(u.len(), NUM_DECISION_VARIABLES, "wrong length of `u`");
    assert_eq!(
        static_params.len(),
        NUM_STATIC_PARAMETERS,
        "wrong length of `u`"
    );
    assert_eq!(
        cost_jacobian.len(),
        NUM_DECISION_VARIABLES,
        "wrong length of `cost_jacobian`"
    );

    let arguments = &[u.as_ptr(), xi.as_ptr(), static_params.as_ptr()];
    let grad = &mut [cost_jacobian.as_mut_ptr()];

    unsafe {
        grad_cost_function_{{ meta.optimizer_name }}(
            arguments.as_ptr(),
            grad.as_mut_ptr()
        ) as i32
    }
}


/// Consume mapping F1, which has been generated by CasADi
///
/// This is a wrapper function
///
/// ## Arguments
///
/// - `u`: (in) decision variables
/// - `p`: (in) vector of parameters
/// - `f1`: (out) value F2(u, p)
///
/// ## Returns
///
/// Returns `0` iff the computation is successful
///
pub fn mapping_f1(
    u: &[f64],
    static_params: &[f64],
    f1: &mut [f64],
) -> i32 {
    assert_eq!(
        u.len(),
        NUM_DECISION_VARIABLES,
        "Incompatible dimension of `u`"
    );
    assert_eq!(
        static_params.len(),
        NUM_STATIC_PARAMETERS,
        "Incompatible dimension of `p`"
    );
    assert!(
        f1.len() == NUM_CONSTRAINTS_TYPE_ALM || NUM_CONSTRAINTS_TYPE_ALM == 0,
        "Incompatible dimension of `f1` (result)"
    );

    let arguments = &[u.as_ptr(), static_params.as_ptr()];
    let constraints = &mut [f1.as_mut_ptr()];

    unsafe {
         mapping_f1_function_{{ meta.optimizer_name }}(
            arguments.as_ptr(),
            constraints.as_mut_ptr()
        ) as i32
    }
}

/// Consume mapping F2, which has been generated by CasADi
///
/// This is a wrapper function
///
/// ## Arguments
///
/// - `u`: (in) decision variables
/// - `p`: (in) vector of parameters
/// - `f2`: (out) value F2(u, p)
///
/// ## Returns
///
/// Returns `0` iff the computation is successful
pub fn mapping_f2(
    u: &[f64],
    static_params: &[f64],
    f2: &mut [f64],
) -> i32 {
    assert_eq!(
        u.len(),
        NUM_DECISION_VARIABLES,
        "Incompatible dimension of `u`"
    );
    assert_eq!(
        static_params.len(),
        NUM_STATIC_PARAMETERS,
        "Incompatible dimension of `p`"
    );
    assert!(
        f2.len() == NUM_CONSTRAINTS_TYPE_PENALTY || NUM_CONSTRAINTS_TYPE_PENALTY == 0,
        "Incompatible dimension of `f2` (result)"
    );

    let arguments = &[u.as_ptr(), static_params.as_ptr()];
    let constraints = &mut [f2.as_mut_ptr()];

    unsafe {
         mapping_f2_function_{{ meta.optimizer_name }}(
            arguments.as_ptr(),
            constraints.as_mut_ptr()
        ) as i32
    }
}

/// Consume mapping Rectangle lower bounds, which has been generated by CasADi
///
/// This is a wrapper function
///
/// ## Arguments
///
/// - `p`: (in) vector of parameters
/// - `bounds`: (out) values bounds U(p)
///
/// ## Returns
///
/// Returns `0` iff the computation is successful
///
pub fn rectangle_lower(
    static_params: &[f64],
    bounds: &mut [f64],
) -> i32 {
    assert_eq!(
        static_params.len(),
        NUM_STATIC_PARAMETERS,
        "Incompatible dimension of `p`"
    );
    assert_eq!(
        bounds.len(),
        NUM_DECISION_VARIABLES,
        "wrong length of `bounds`"
    );

    let arguments = &[static_params.as_ptr()];
    let constraints = &mut [bounds.as_mut_ptr()];

    unsafe {
         rectangle_lower_function_{{ meta.optimizer_name }}(
            arguments.as_ptr(),
            constraints.as_mut_ptr()
        ) as i32
    }
}

/// Consume mapping Rectangle upper bounds, which has been generated by CasADi
///
/// This is a wrapper function
///
/// ## Arguments
///
/// - `p`: (in) vector of parameters
/// - `bounds`: (out) values bounds U(p)
///
/// ## Returns
///
/// Returns `0` iff the computation is successful
///
pub fn rectangle_upper(
    static_params: &[f64],
    bounds: &mut [f64],
) -> i32 {
    assert_eq!(
        static_params.len(),
        NUM_STATIC_PARAMETERS,
        "Incompatible dimension of `p`"
    );
    assert_eq!(
        bounds.len(),
        NUM_DECISION_VARIABLES,
        "wrong length of `bounds`"
    );

    let arguments = &[static_params.as_ptr()];
    let constraints = &mut [bounds.as_mut_ptr()];

    unsafe {
         rectangle_upper_function_{{ meta.optimizer_name }}(
            arguments.as_ptr(),
            constraints.as_mut_ptr()
        ) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tst_call_cost() {
        let u = [0.1; NUM_DECISION_VARIABLES];
        let p = [0.1; NUM_STATIC_PARAMETERS];
        let xi = [2.0; NUM_CONSTRAINTS_TYPE_ALM+1];
        let mut cost = 0.0;
        assert_eq!(0, super::cost(&u, &xi, &p, &mut cost));
    }

    #[test]
    fn tst_call_grad() {
        let u = [0.1; NUM_DECISION_VARIABLES];
        let p = [0.1; NUM_STATIC_PARAMETERS];
        let xi = [10.0; NUM_CONSTRAINTS_TYPE_ALM+1];
        let mut grad = [0.0; NUM_DECISION_VARIABLES];
        assert_eq!(0, super::grad(&u, &xi, &p, &mut grad));
    }

    #[test]
    fn tst_f1() {
        let u = [0.1; NUM_DECISION_VARIABLES];
        let p = [0.1; NUM_STATIC_PARAMETERS];
        let mut f1up = [0.0; NUM_CONSTRAINTS_TYPE_ALM];
        assert_eq!(0, super::mapping_f1(&u, &p, &mut f1up));
    }

    #[test]
    fn tst_f2() {
        let u = [0.1; NUM_DECISION_VARIABLES];
        let p = [0.1; NUM_STATIC_PARAMETERS];
        let mut f2up = [0.0; NUM_CONSTRAINTS_TYPE_PENALTY];
        assert_eq!(0, super::mapping_f2(&u, &p, &mut f2up));
    }

}

