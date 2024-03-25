"""ODE integrator classes for timestepping."""

from collections import deque, OrderedDict
import numpy as np
from scipy.linalg import blas

from .system import CoeffSystem
from ..tools.array import apply_sparse


# Public interface
__all__ = []

# Track implemented schemes
schemes = OrderedDict()
def add_scheme(scheme):
    __all__.append(scheme.__name__)
    schemes[scheme.__name__] = scheme
    return scheme


class MultistepIMEX:
    """
    Base class for implicit-explicit multistep methods.

    Parameters
    ----------
    nfields : int
        Number of fields in problem
    domain : domain object
        Problem domain

    Notes
    -----
    These timesteppers discretize the system
        M.dt(X) + L.X = F
    into the general form
        aj M.X(n-j) + bj L.X(n-j) = cj F(n-j)
    where j runs from {0, 0, 1} to {amax, bmax, cmax}.

    The system is then solved as
        (a0 M + b0 L).X(n) = cj F(n-j) - aj M.X(n-j) - bj L.X(n-j)
    where j runs from {1, 1, 1} to {cmax, amax, bmax}.

    References
    ----------
    D. Wang and S. J. Ruuth, Journal of Computational Mathematics 26, (2008).*

    * Our coefficients are related to those used by Wang as:
        amax = bmax = cmax = s
        aj = α(s-j) / k(n+s-1)
        bj = γ(s-j)
        cj = β(s-j)

    """

    stages = 1

    def __init__(self, solver):

        self.solver = solver
        self.RHS = CoeffSystem(solver.subproblems, dtype=solver.dtype)

        self.sensRHS = CoeffSystem(solver.subproblems, dtype=solver.dtype)

        # Create deque for storing recent timesteps
        self.dt = deque([0.] * self.steps)

        # Create coefficient systems for multistep history
        self.MX = MX = deque()
        self.LX = LX = deque()
        self.F = F = deque()

        self.sens = sens = deque()
        for j in range(self.amax):
            MX.append(CoeffSystem(solver.subproblems, dtype=solver.dtype))
        for j in range(self.bmax):
            LX.append(CoeffSystem(solver.subproblems, dtype=solver.dtype))
        for j in range(self.cmax):
            F.append(CoeffSystem(solver.subproblems, dtype=solver.dtype))
        for j in range(self.cmax):
            sens.append(CoeffSystem(solver.subproblems, dtype=solver.dtype))

        # For the adjoint
        self.timestep_history = []
        # Attributes
        self._iteration = 0
        self._LHS_params = None
        self.axpy = blas.get_blas_funcs('axpy', dtype=solver.dtype)

    def step(self, dt, wall_time):
        """Advance solver by one timestep."""

        # Solver references
        solver = self.solver
        subproblems = [sp for sp in solver.subproblems if sp.size] # Skip empty subproblems
        evaluator = solver.evaluator
        state_fields = solver.state
        F_fields = solver.F
        sim_time = solver.sim_time
        iteration = solver.iteration
        STORE_EXPANDED_MATRICES = solver.store_expanded_matrices

        # Other references
        MX = self.MX
        LX = self.LX
        F = self.F
        RHS = self.RHS
        axpy = self.axpy

        # Cycle and compute timesteps
        self.dt.rotate()
        self.dt[0] = dt
        # For the adjoint solve
        self.timestep_history.append(dt)

        # Compute IMEX coefficients
        a, b, c = self.compute_coefficients(self.dt, self._iteration)
        self._iteration += 1

        # Update RHS components and LHS matrices
        MX.rotate()
        LX.rotate()
        F.rotate()

        MX0 = MX[0]
        LX0 = LX[0]
        F0 = F[0]
        a0 = a[0]
        b0 = b[0]

        # Check on updating LHS
        update_LHS = ((a0, b0) != self._LHS_params)
        self._LHS_params = (a0, b0)
        if update_LHS:
            # Remove old solver references
            for sp in subproblems:
                sp.LHS_solver = None

        # Evaluate M.X0 and L.X0
        evaluator.require_coeff_space(state_fields)
        for sp in subproblems:
            spX = sp.gather_inputs(state_fields)
            apply_sparse(sp.M_min, spX, axis=0, out=MX0.get_subdata(sp))
            apply_sparse(sp.L_min, spX, axis=0, out=LX0.get_subdata(sp))

        # Evaluate F(X0)
        evaluator.evaluate_scheduled(iteration=iteration, wall_time=wall_time, sim_time=sim_time, timestep=dt)
        evaluator.require_coeff_space(F_fields)
        for sp in subproblems:
            sp.gather_outputs(F_fields, out=F0.get_subdata(sp))

        # Build RHS
        if RHS.data.size:
            np.multiply(c[1], F0.data, out=RHS.data)
            for j in range(2, len(c)):
                # RHS.data += c[j] * F[j-1].data
                axpy(a=c[j], x=F[j-1].data, y=RHS.data)
            for j in range(1, len(a)):
                # RHS.data -= a[j] * MX[j-1].data
                axpy(a=-a[j], x=MX[j-1].data, y=RHS.data)
            for j in range(1, len(b)):
                # RHS.data -= b[j] * LX[j-1].data
                axpy(a=-b[j], x=LX[j-1].data, y=RHS.data)

        # Solve
        # Ensure coeff space before subsystem scatters
        for field in state_fields:
            field.preset_layout('c')
        for sp in subproblems:
            if update_LHS:
                if STORE_EXPANDED_MATRICES:
                    # sp.LHS.data[:] = a0*sp.M_exp.data + b0*sp.L_exp.data
                    np.multiply(a0, sp.M_exp.data, out=sp.LHS.data)
                    axpy(a=b0, x=sp.L_exp.data, y=sp.LHS.data)
                else:
                    sp.LHS = (a0*sp.M_min + b0*sp.L_min)  # CREATES TEMPORARY
                sp.LHS_solver = solver.matsolver(sp.LHS, solver)
            # Slice out valid subdata, skipping invalid components
            spRHS = RHS.get_subdata(sp)
            spX = sp.LHS_solver.solve(spRHS)  # CREATES TEMPORARY
            sp.scatter_inputs(spX, state_fields)

        # Update solver
        solver.sim_time += dt

    def step_adjoint(self, dt, wall_time):
        """Advance adjoint by one timestep."""

        # Solver references
        solver = self.solver
        subproblems = solver.subproblems
        evaluator = solver.evaluator
        state_fields = solver.state_adj
        sens_fields = solver.sens_adj
        F_fields = solver.F
        sim_time = solver.sim_time
        iteration = solver.iteration
        STORE_EXPANDED_MATRICES = solver.store_expanded_matrices

        # Other references
        MX = self.MX
        LX = self.LX
        F = self.F
        sens=self.sens

        RHS = self.RHS
        sensRHS = self.sensRHS
        axpy = self.axpy

        self._iteration -= 1
        # dt = self.timestep_history[self._iteration]

        # This part of the code could probably be tidier
        # Compute IMEX coefficients at current iteration and up to steps-1 iterations in the future
        a = []
        b = []
        c = []
        for k in range(self.steps):
            # If the future steps aren't beyond the last iteration
            if(self._iteration+k < solver.stop_iteration):
                # Compute the dt deque at iteration self._iteration+k
                # Do not fill the deque with entries not present for initial start up iterations
                # (put zero instead)
                max_steps = np.min([self.steps,self._iteration+k+1])
                for step in range(self.steps):
                    if(step<max_steps):
                        self.dt[step] = self.timestep_history[self._iteration+k-step]
                    else:
                        self.dt[step] = 0

                # Compute a,b,c at iteration self._iteration+k
                a_, b_, c_ = self.compute_coefficients(self.dt, self._iteration+k)
                a.append(a_)
                b.append(b_)
                c.append(c_)

        
        # Update RHS components and LHS matrices

        MX.rotate()
        LX.rotate()
        F.rotate()
        sens.rotate()

        MX0 = MX[0]
        LX0 = LX[0]
        F0 = F[0]
        sens0 = sens[0]

        a0 = a[0][0]
        b0 = b[0][0]

        # Check on updating LHS
        if(solver.iteration==solver.stop_iteration):
            # Must clear the deque
            for m in MX:
                m.data.fill(0)
            for l in LX:
                l.data.fill(0)
            for f in F:
                f.data.fill(0)

        update_LHS = ((a0, b0) != self._LHS_params) or solver.iteration==solver.stop_iteration
        self._LHS_params = (a0, b0)

        MX0.data.fill(0)
        LX0.data.fill(0)
        F0.data.fill(0)

        # Solve, form L, M, F terms, then form next RHS
        # Ensure coeff space before subsystem gathers
      
        for sp in subproblems:
            if update_LHS:
                # Remove old solver reference
                sp.LHS_solver = None
            if update_LHS:
                if STORE_EXPANDED_MATRICES:
                    # sp.LHS.data[:] = a0*sp.M_exp.data + b0*sp.L_exp.data
                    np.multiply(a0, sp.M_exp.data, out=sp.LHS.data)
                    axpy(a=b0, x=sp.L_exp.data, y=sp.LHS.data)
                else:
                    sp.LHS = (a0*sp.M_min + b0*sp.L_min)  # CREATES TEMPORARY
                sp.LHS_solver = solver.matsolver(np.conj(sp.LHS).T.tocsr(), solver)

        # Ensure coeff space before subsystem gathers
        for field in state_fields:
            field.require_coeff_space()
     
        for sp in subproblems:
            # Slice out valid subdata, skipping invalid components

            spRHS = sp.gather_inputs(state_fields)

            spX = sp.LHS_solver.solve(spRHS)  # CREATES TEMPORARY
            # TODO: Do something better for the csr conversion
            apply_sparse(np.conj(sp.M_min).T.tocsr(), spX, axis=0, out=MX0.get_subdata(sp))  # Rectangular dot product skipping shape checks
            apply_sparse(np.conj(sp.L_min).T.tocsr(), spX, axis=0, out=LX0.get_subdata(sp))  # Rectangular dot product skipping shape checks

            sp.scatter_inputs(spX, state_fields) # 1

        # evaluator.evaluate_scheduled(wall_time=wall_time, timestep=dt, sim_time=solver.sim_time, iteration=iteration)
        if hasattr(solver,'F_adjoint'):
            evaluator.evaluate_group('F_adjoint', wall_time=wall_time, timestep=dt, sim_time=solver.sim_time, iteration=iteration)
            for sp in subproblems:
                sp.gather_outputs(solver.F_adjoint, out=F0.get_subdata(sp)) 
        if hasattr(solver,'F_sens'):
            evaluator.evaluate_group('F_sens', wall_time=wall_time, timestep=dt, sim_time=solver.sim_time, iteration=iteration)
            for sp in subproblems:
                sp.gather_outputs(solver.F_sens, out=sens0.get_subdata(sp))

        

        if RHS.data.size:
            np.multiply(c[0][1], F0.data, out=RHS.data)
            for j in range(2, len(c) + 1):
                # RHS.data += c[j] * F[j-1].data
                axpy(a=c[j-1][j], x=F[j-1].data, y=RHS.data)
            for j in range(1, len(a)+1):
                # RHS.data -= a[j] * MX[j-1].data
                axpy(a=-a[j-1][j], x=MX[j-1].data, y=RHS.data)
            for j in range(1, len(b)+1):
                # RHS.data -= b[j] * LX[j-1].data
                axpy(a=-b[j-1][j], x=LX[j-1].data, y=RHS.data)
        
        for field in state_fields:
            field.preset_layout('c')
        for sp in subproblems:
            sp.scatter_inputs(RHS.get_subdata(sp), state_fields)

        if hasattr(solver,'F_sens'):
            if sensRHS.data.size:
                # np.multiply(c[0][1], sens0.data, out=sensRHS.data)
                for j in range(1, len(c) + 1):
                    # RHS.data += c[j] * F[j-1].data
                    axpy(a=c[j-1][j], x=sens[j-1].data, y=sensRHS.data)
            for field in sens_fields:
                field.preset_layout('c')
            for sp in subproblems:
                sp.scatter_inputs(sensRHS.get_subdata(sp), sens_fields)
            
        # Update solver
        solver.sim_time -= dt


@add_scheme
class CNAB1(MultistepIMEX):
    """
    1st-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.5.3]

    Implicit: 2nd-order Crank-Nicolson
    Explicit: 1st-order Adams-Bashforth (forward Euler)

    """

    amax = 1
    bmax = 1
    cmax = 1
    steps = 1

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k0, *rest = timesteps

        a[0] = 1 / k0
        a[1] = -1 / k0
        b[0] = 1 / 2
        b[1] = 1 / 2
        c[1] = 1

        return a, b, c


@add_scheme
class SBDF1(MultistepIMEX):
    """
    1st-order semi-implicit BDF scheme [Wang 2008 eqn 2.6]

    Implicit: 1st-order BDF (backward Euler)
    Explicit: 1st-order extrapolation (forward Euler)

    """

    amax = 1
    bmax = 1
    cmax = 1
    steps = 1

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k0, *rest = timesteps

        a[0] = 1 / k0
        a[1] = -1 / k0
        b[0] = 1
        c[1] = 1

        return a, b, c


@add_scheme
class CNAB2(MultistepIMEX):
    """
    2nd-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.9]

    Implicit: 2nd-order Crank-Nicolson
    Explicit: 2nd-order Adams-Bashforth

    """

    amax = 2
    bmax = 2
    cmax = 2
    steps = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return CNAB1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = 1 / 2
        b[1] = 1 / 2
        c[1] = 1 + w1/2
        c[2] = -w1 / 2

        return a, b, c


@add_scheme
class MCNAB2(MultistepIMEX):
    """
    2nd-order modified Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.10]

    Implicit: 2nd-order modified Crank-Nicolson
    Explicit: 2nd-order Adams-Bashforth

    """

    amax = 2
    bmax = 2
    cmax = 2
    steps = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return CNAB1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = (8 + 1/w1) / 16
        b[1] = (7 - 1/w1) / 16
        b[2] = 1 / 16
        c[1] = 1 + w1/2
        c[2] = -w1 / 2

        return a, b, c


@add_scheme
class SBDF2(MultistepIMEX):
    """
    2nd-order semi-implicit BDF scheme [Wang 2008 eqn 2.8]

    Implicit: 2nd-order BDF
    Explicit: 2nd-order extrapolation

    """

    amax = 2
    bmax = 2
    cmax = 2
    steps = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return SBDF1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = (1 + 2*w1) / (1 + w1) / k1
        a[1] = -(1 + w1) / k1
        a[2] = w1**2 / (1 + w1) / k1
        b[0] = 1
        c[1] = 1 + w1
        c[2] = -w1

        return a, b, c


@add_scheme
class CNLF2(MultistepIMEX):
    """
    2nd-order Crank-Nicolson leap-frog scheme [Wang 2008 eqn 2.11]

    Implicit: ?-order wide Crank-Nicolson
    Explicit: 2nd-order leap-frog

    """

    amax = 2
    bmax = 2
    cmax = 2
    steps = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return CNAB1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / (1 + w1) / k1
        a[1] = (w1 - 1) / k1
        a[2] = -w1**2 / (1 + w1) / k1
        b[0] = 1 / w1 / 2
        b[1] = (1 - 1/w1) / 2
        b[2] = 1 / 2
        c[1] = 1

        return a, b, c


@add_scheme
class SBDF3(MultistepIMEX):
    """
    3rd-order semi-implicit BDF scheme [Wang 2008 eqn 2.14]

    Implicit: 3rd-order BDF
    Explicit: 3rd-order extrapolation

    """

    amax = 3
    bmax = 3
    cmax = 3
    steps = 3

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 2:
            return SBDF2.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k2, k1, k0, *rest = timesteps
        w2 = k2 / k1
        w1 = k1 / k0

        a[0] = (1 + w2/(1 + w2) + w1*w2/(1 + w1*(1 + w2))) / k2
        a[1] = (-1 - w2 - w1*w2*(1 + w2)/(1 + w1)) / k2
        a[2] = w2**2 * (w1 + 1/(1 + w2)) / k2
        a[3] = -w1**3 * w2**2 * (1 + w2) / (1 + w1) / (1 + w1 + w1*w2) / k2
        b[0] = 1
        c[1] = (1 + w2)*(1 + w1*(1 + w2)) / (1 + w1)
        c[2] = -w2*(1 + w1*(1 + w2))
        c[3] = w1*w1*w2*(1 + w2) / (1 + w1)

        return a, b, c


@add_scheme
class SBDF4(MultistepIMEX):
    """
    4th-order semi-implicit BDF scheme [Wang 2008 eqn 2.15]

    Implicit: 4th-order BDF
    Explicit: 4th-order extrapolation

    """

    amax = 4
    bmax = 4
    cmax = 4
    steps = 4

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 3:
            return SBDF3.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k3, k2, k1, k0, *rest = timesteps
        w3 = k3 / k2
        w2 = k2 / k1
        w1 = k1 / k0

        A1 = 1 + w1*(1 + w2)
        A2 = 1 + w2*(1 + w3)
        A3 = 1 + w1*A2

        a[0] = (1 + w3/(1 + w3) + w2*w3/A2 + w1*w2*w3/A3) / k3
        a[1] = (-1 - w3*(1 + w2*(1 + w3)/(1 + w2)*(1 + w1*A2/A1))) / k3
        a[2] = w3 * (w3/(1 + w3) + w2*w3*(A3 + w1)/(1 + w1)) / k3
        a[3] = -w2**3 * w3**2 * (1 + w3) / (1 + w2) * A3 / A2 / k3
        a[4] = (1 + w3) / (1 + w1) * A2 / A1 * w1**4 * w2**3 * w3**2 / A3 / k3
        b[0] = 1
        c[1] = w2 * (1 + w3) / (1 + w2) * ((1 + w3)*(A3 + w1) + (1 + w1)/w2) / A1
        c[2] = -A2 * A3 * w3 / (1 + w1)
        c[3] = w2**2 * w3 * (1 + w3) / (1 + w2) * A3
        c[4] = -w1**3 * w2**2 * w3 * (1 + w3) / (1 + w1) * A2 / A1

        return a, b, c


class RungeKuttaIMEX:
    """
    Base class for implicit-explicit multistep methods.

    Parameters
    ----------
    nfields : int
        Number of fields in problem
    domain : domain object
        Problem domain

    Notes
    -----
    These timesteppers discretize the system
        M.dt(X) + L.X = F
    by constructing s stages
        M.X(n,i) - M.X(n,0) + k Hij L.X(n,j) = k Aij F(n,j)
    where j runs from {0, 0} to {i, i-1}, and F(n,i) is evaluated at time
        t(n,i) = t(n,0) + k ci

    The s stages are solved as
        (M + k Hii L).X(n,i) = M.X(n,0) + k Aij F(n,j) - k Hij L.X(n,j)
    where j runs from {0, 0} to {i-1, i-1}.

    The final stage is used as the advanced solution*:
        X(n+1,0) = X(n,s)
        t(n+1,0) = t(n,s) = t(n,0) + k

    * Equivalently the Butcher tableaus must follow
        b_im = H[s, :]
        b_ex = A[s, :]
        c[s] = 1

    References
    ----------
    U. M. Ascher, S. J. Ruuth, and R. J. Spiteri, Applied Numerical Mathematics (1997).

    """

    steps = 1

    def __init__(self, solver):

        self.solver = solver
        self.RHS = CoeffSystem(solver.subproblems, dtype=solver.dtype)

        # Create coefficient systems for multistep history
        self.MX0 = CoeffSystem(solver.subproblems, dtype=solver.dtype)
        self.LX = [CoeffSystem(solver.subproblems, dtype=solver.dtype) for i in range(self.stages)]
        self.F = [CoeffSystem(solver.subproblems, dtype=solver.dtype) for i in range(self.stages)]

        self._LHS_params = None
        self.axpy = blas.get_blas_funcs('axpy', dtype=solver.dtype)

        # For adjoint for now
        self.MXT = [CoeffSystem(solver.subproblems, dtype=solver.dtype) for i in range(self.stages)]
        
        self.timestep_history = []

    def step(self, dt, wall_time):
        """Advance solver by one timestep."""

        # Solver references
        solver = self.solver
        subproblems = [sp for sp in solver.subproblems if sp.size] # Skip empty subproblems
        evaluator = solver.evaluator
        state_fields = solver.state
        F_fields = solver.F
        sim_time_0 = solver.sim_time
        iteration = solver.iteration
        STORE_EXPANDED_MATRICES = solver.store_expanded_matrices

        # Other references
        RHS = self.RHS
        MX0 = self.MX0
        LX = self.LX
        LX0 = LX[0]
        F = self.F
        A = self.A
        H = self.H
        c = self.c
        k = dt
        axpy = self.axpy

        # Save dt for adjoint
        self.timestep_history.append(dt)

        # Check on updating LHS
        update_LHS = (k != self._LHS_params)
        # print(self._LHS_params,update_LHS)
        self._LHS_params = k
        if update_LHS:
            # Remove old solver references
            for sp in subproblems:
                sp.LHS_solvers = [None] * (self.stages+1)

        # Compute M.X(n,0) and L.X(n,0)
        # Ensure coeff space before subsystem gathers
        evaluator.require_coeff_space(state_fields)
        for sp in subproblems:
            spX = sp.gather_inputs(state_fields)
            apply_sparse(sp.M_min, spX, axis=0, out=MX0.get_subdata(sp))
            apply_sparse(sp.L_min, spX, axis=0, out=LX0.get_subdata(sp))

        # Compute stages
        # (M + k Hii L).X(n,i) = M.X(n,0) + k Aij F(n,j) - k Hij L.X(n,j)
        for i in range(1, self.stages+1):

            # Compute L.X(n,i-1), already done for i=1
            if i > 1:
                LXi = LX[i-1]
                # Ensure coeff space before subsystem gathers
                evaluator.require_coeff_space(state_fields)
                for sp in subproblems:
                    spX = sp.gather_inputs(state_fields)
                    apply_sparse(sp.L_min, spX, axis=0, out=LXi.get_subdata(sp))

            # Compute F(n,i-1), only doing output on first evaluation
            if i == 1:
                evaluator.evaluate_scheduled(iteration=iteration, wall_time=wall_time, sim_time=solver.sim_time, timestep=dt)
            else:
                evaluator.evaluate_group('F')
            Fi = F[i-1]
            for sp in subproblems:
                # F fields should be in coeff space from evaluator
                sp.gather_outputs(F_fields, out=Fi.get_subdata(sp))

            # Construct RHS(n,i)
            if RHS.data.size:
                np.copyto(RHS.data, MX0.data)
                for j in range(0, i):
                    # RHS.data += (k * A[i,j]) * F[j].data
                    axpy(a=(k*A[i,j]), x=F[j].data, y=RHS.data)
                    # RHS.data -= (k * H[i,j]) * LX[j].data
                    axpy(a=-(k*H[i,j]), x=LX[j].data, y=RHS.data)

            # Solve for stage
            k_Hii = k * H[i,i]
            # Ensure coeff space before subsystem scatters
            for field in state_fields:
                field.preset_layout('c')
            for sp in subproblems:
                # Construct LHS(n,i)
                if update_LHS:
                    if STORE_EXPANDED_MATRICES:
                        # sp.LHS.data[:] = sp.M_exp.data + k_Hii*sp.L_exp.data
                        np.copyto(sp.LHS.data, sp.M_exp.data)
                        axpy(a=k_Hii, x=sp.L_exp.data, y=sp.LHS.data)
                    else:
                        sp.LHS = (sp.M_min + k_Hii*sp.L_min)  # CREATES TEMPORARY
                    sp.LHS_solvers[i] = solver.matsolver(sp.LHS, solver)
                # Slice out valid subdata, skipping invalid components
                spRHS = RHS.get_subdata(sp)
                spX = sp.LHS_solvers[i].solve(spRHS)  # CREATES TEMPORARY
                sp.scatter_inputs(spX, state_fields)
            solver.sim_time = sim_time_0 + k*c[i]

    def step_adjoint(self, dt, wall_time):
        """Advance solver by one timestep."""

        # Solver references
        solver = self.solver
        subproblems = solver.subproblems
        evaluator = solver.evaluator
        state_fields = solver.state_adj
        F_fields = solver.F
        sim_time_0 = solver.sim_time
        iteration = solver.iteration
        STORE_EXPANDED_MATRICES = solver.store_expanded_matrices

        # Other references
        RHS = self.RHS
        MX0 = self.MX0
        LX = self.LX
        F = self.F
        A = self.A
        H = self.H
        c = self.c
        # k = dt = self.timestep_history[solver.iteration-1]
        k = dt
        axpy = self.axpy

        MXT = self.MXT

        # Check on updating LHS
        update_LHS = solver.iteration==solver.stop_iteration or (k != self._LHS_params)
        
        self._LHS_params = k

        for sp in subproblems:
            if update_LHS:
                # Remove old solver references
                sp.LHS_solvers = [None] * (self.stages+1)

        # Final time condition
        # Ensure coeff space before subsystem gathers
        for field in state_fields:
            field.require_coeff_space()
        # Ensure coeff space before subsystem gathers
        evaluator.require_coeff_space(state_fields)

        RHS.data.fill(0)
        for sp in subproblems:
            # Use adjoint state for RHS
            np.copyto(RHS.get_subdata(sp),sp.gather_inputs(state_fields))
        
        # Compute stages
        for i in reversed(range(1, self.stages+1)):
            if(i<self.stages):
                # Ensure coeff space before subsystem gathers
                evaluator.require_coeff_space(state_fields)
            # Solve for stage
    
            # Clear coeff system for transposed data
            LXi = LX[i-1]
            LXi.data.fill(0)

            Fi = F[i-1]
            Fi.data.fill(0)

            MXTi = MXT[i-1]
            MXTi.data.fill(0)

            # WARNING: Not adjointed yet (assumed F is zero)
            # Compute F(n,i-1) (needs to be adjointed here...)            
            k_Hii = k * H[i,i]

            for field in state_fields:
                field.preset_layout('c')
            for sp in subproblems:
                # Construct LHS(n,i)
                if update_LHS: # just for now
                    if STORE_EXPANDED_MATRICES:
                        # sp.LHS.data[:] = sp.M_exp.data + k_Hii*sp.L_exp.data
                        np.copyto(sp.LHS.data, sp.M_exp.data)
                        axpy(a=k_Hii, x=sp.L_exp.data, y=sp.LHS.data)
                    else:
                        sp.LHS = (sp.M_min + k_Hii*sp.L_min)  # CREATES TEMPORARY

                    # Create transposed solver
                    sp.LHS_solvers[i] = solver.matsolver(np.conj(sp.LHS).T, solver)

                # Use computed adjoint RHS
                spRHS = RHS.get_subdata(sp)

                spX = sp.LHS_solvers[i].solve(spRHS)  # CREATES TEMPORARY

                # Compute new transpose terms and RHS
                # TODO: Do something better for the csr conversion
                apply_sparse((np.conj(sp.L_min).T).tocsr(), spX, axis=0, out=LXi.get_subdata(sp))  # Rectangular dot product skipping shape checks                
                apply_sparse((np.conj(sp.M_min).T).tocsr(), spX, axis=0, out=MXTi.get_subdata(sp))
                # print(F_fields)
                # RHS similar but not implemented yet
                # evaluator.evaluate_group('F_adjoint')
                sp.scatter_inputs(spX, state_fields) # 1

            # if i == self.stages:
                # evaluator.evaluate_scheduled(wall_time=wall_time, timestep=dt, sim_time=solver.sim_time, iteration=iteration)
                # evaluator.evaluate_group('F_adjoint', wall_time=wall_time, timestep=dt, sim_time=solver.sim_time, iteration=iteration)
            # else:
            
            if hasattr(solver,'F_adjoint'):
                evaluator.evaluate_group('F_adjoint', wall_time=wall_time, timestep=dt, sim_time=solver.sim_time, iteration=iteration)
            
            # for f in solver.F_adjoint:
            #     print(f.layout.index)
                for sp in subproblems:
                    sp.gather_outputs(solver.F_adjoint, out=Fi.get_subdata(sp)) # 1

                # csr_matvecs(np.conj(sp.pre_right).T.tocsr()np.conj(sp.pre_left).T.tocsr(), spX, Fi.get_subdata(sp))  # Rectangular dot product skipping shape checks             
                # Compute adjoint RHS

            if RHS.data.size:
                RHS.data.fill(0)
                for j in range(i, self.stages+1):
                    # RHS.data += (k * A[j,i]) * FT[j-1].data
                    axpy(a=(k*A[j,i-1]), x=F[j-1].data, y=RHS.data)
                    # RHS.data -= (k * H[j,i]) * LXT[j].data
                    axpy(a=-(k*H[j,i-1]), x=LX[j-1].data, y=RHS.data)
               
            solver.sim_time = sim_time_0 - k + k*c[i-1]

        # At last stage add on MXT terms
        if RHS.data.size:
            for j in range(1, self.stages+1):
                # RHS.data += (1 * H[j,0]) * MXT[j].data
                axpy(a=1, x=MXT[j-1].data, y=RHS.data)

        # Ensure coeff space before subsystem scatters
        for field in state_fields:
            field.preset_layout('c')

        for sp in subproblems:
            # Adjoint state is the RHS (Use pre-right to undo pre-conditioned RHS)        
            spRHS = RHS.get_subdata(sp)
            sp.scatter_inputs(spRHS, state_fields)

@add_scheme
class RK111(RungeKuttaIMEX):
    """1st-order 1-stage DIRK+ERK scheme [Ascher 1997 sec 2.1]"""

    stages = 1

    c = np.array([0, 1])

    A = np.array([[0, 0],
                  [1, 0]])

    H = np.array([[0, 0],
                  [0, 1]])


@add_scheme
class RK222(RungeKuttaIMEX):
    """2nd-order 2-stage DIRK+ERK scheme [Ascher 1997 sec 2.6]"""

    stages = 2

    γ = (2 - np.sqrt(2)) / 2
    δ = 1 - 1 / γ / 2

    c = np.array([0, γ, 1])

    A = np.array([[0,  0 , 0],
                  [γ,  0 , 0],
                  [δ, 1-δ, 0]])

    H = np.array([[0,  0 , 0],
                  [0,  γ , 0],
                  [0, 1-γ, γ]])


@add_scheme
class RK443(RungeKuttaIMEX):
    """3rd-order 4-stage DIRK+ERK scheme [Ascher 1997 sec 2.8]"""

    stages = 4

    c = np.array([0, 1/2, 2/3, 1/2, 1])

    A = np.array([[  0  ,   0  ,  0 ,   0 , 0],
                  [ 1/2 ,   0  ,  0 ,   0 , 0],
                  [11/18,  1/18,  0 ,   0 , 0],
                  [ 5/6 , -5/6 , 1/2,   0 , 0],
                  [ 1/4 ,  7/4 , 3/4, -7/4, 0]])

    H = np.array([[0,   0 ,   0 ,  0 ,  0 ],
                  [0,  1/2,   0 ,  0 ,  0 ],
                  [0,  1/6,  1/2,  0 ,  0 ],
                  [0, -1/2,  1/2, 1/2,  0 ],
                  [0,  3/2, -3/2, 1/2, 1/2]])


@add_scheme
class RKSMR(RungeKuttaIMEX):
    """(3-ε)-order 3rd-stage DIRK+ERK scheme [Spalart 1991 Appendix]"""

    stages = 3

    α1, α2, α3 = (29/96, -3/40, 1/6)
    β1, β2, β3 = (37/160, 5/24, 1/6)
    γ1, γ2, γ3 = (8/15, 5/12, 3/4)
    ζ2, ζ3 = (-17/60, -5/12)

    c = np.array([0, 8/15, 2/3, 1])

    A = np.array([[    0,     0,  0, 0],
                  [   γ1,     0,  0, 0],
                  [γ1+ζ2,    γ2,  0, 0],
                  [γ1+ζ2, γ2+ζ3, γ3, 0]])

    H = np.array([[ 0,     0,     0,  0],
                  [α1,    β1,     0,  0],
                  [α1, β1+α2,    β2,  0],
                  [α1, β1+α2, β2+α3, β3]])


class RKGFY(RungeKuttaIMEX):
    """2nd-order 2-stage scheme from Hollerbach and Marti"""

    stages = 2

    c = np.array([0, 1, 1])

    A = np.array([[  0,  0 , 0],
                  [  1,  0 , 0],
                  [0.5, 0.5, 0]])

    H = np.array([[0   , 0  ,   0],
                  [0.5 , 0.5,   0],
                  [0.5 , 0  , 0.5]])

