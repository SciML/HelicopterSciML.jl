using Pkg
cd(@__DIR__)
Pkg.activate("../"); Pkg.instantiate()

using Plots; gr()
using LaTeXStrings
using DataFrames
using CSV
using OrdinaryDiffEq
using LinearAlgebra
using DataInterpolations
using BenchmarkTools
using DiffEqFlux
using Optim
using Flux
using Zygote
using ModelingToolkit
using DataDrivenDiffEq

model_univ = FastChain(FastDense(6, 16, tanh),
                       FastDense(16, 2))
include("../optimization_results/neural_augmentation_results_ux.jl")
include("../optimization_results/globaloptimizationresults.jl")

# Plot path
figpath = "../figs/";
# Linewidths and styles
LW1 = 2.5
LW2 = 1.5
LS1 = :solid
LS2 = :dot
LS3 = :dash
LC1 = :blue
LC2 = :red
LC3 = :green
LC4 = :black
LA1 = 1.0
LA2 = 0.7
LA3 = 0.3
;

df=CSV.read("../data/Lab-Helicopter_Experimental-data.csv")
ENV["COLUMNS"]=100 # change the number of IJulia columns from default 80 to 100
df[1:5,:]

u_th = df[:,1]
u_psi = df[:,2]
theta = df[:,3]
psi = df[:,4]
dt = df[1,5]
tm = 0:dt:(length(psi)-1)*dt
;

timeseriesdata = [theta';psi']
Nd = length(tm);

U_th = ConstantInterpolation(u_th,tm)
U_psi = ConstantInterpolation(u_psi,tm)
Theta = ConstantInterpolation(theta,tm)
Psi = ConstantInterpolation(psi,tm);
u_t = (th=U_th, psi=U_psi)
rad2deg2(t,u) = (t,u*180/pi)

#### Sparsify

# Create a Basis
@variables u[1:6]
# Lots of polynomials
polys = Operation[]

for i ∈ 0:2, j ∈ 0:2, k ∈ 0:2, l ∈ 0:2, m ∈ 0:2, n ∈ 0:2
    s = i + j + k + l + m + n
    s <= 2 && push!(polys, u[1]^i * u[2]^j * u[3]^k * u[4]^l * u[5]^m * u[6]^n)
end
#=
for k ∈ 0:2, l ∈ 0:2, m ∈ 0:2, n ∈ 0:2
    push!(polys, u[3]^k * u[4]^l * u[5]^m * u[6]^n)
end


# And some other stuff
h = [cos.(u)...; sin.(u)...; unique(polys)...]
basis_ux = Basis(h, u)
=#

# And some other stuff
h = [cos.(polys)...; sin.(polys)...; polys...]
basis_ux = Basis(h, u)
println(basis_ux)
# Create an optimizer for the SINDY problem
opt = SR3()
# Create the thresholds which should be used in the search process
λ = exp10.(-10:0.1:10)
# Target function to choose the results from; x = L0 of coefficients and L2-Error of the model
f_target(x, w) = iszero(x[1]) ? Inf : norm(w.*x, 2)

# Test on original data and without further knowledge
X = reduce(hcat,utimeseries)
DX = reduce(hcat,Ktimeseries)
Ψ = SInDy(X, DX, basis_ux, λ, opt = opt, maxiter = 10000, f_target = f_target)
println(Ψ)
print_equations(Ψ)

###########
## Check the fit
###########

function helicopter_uode(dx,x,Ps,P0,u,p,t)
# P0 = (rc=rc,h=h,m=m,Jbc=Jbc,Js=Js,g=g,K_th_th=K_th_th,K_th_psi=K_th_psi,K_psi_th=K_psi_th,K_psi_psi=K_psi_psi,D_th=D_th,D_psi=D_psi);
    th = x[1]
    psi = x[2]
    w_th = x[3]
    w_psi = x[4]

    U = [u.th(t),u.psi(t)]
    K = Ψ.equations.f_([U;x],Ψ.parameters,t)

    #
    M11 = Ps[4]*P0.Jbc + Ps[3]*P0.m*((Ps[1]*P0.rc)^2+(Ps[2]*P0.h)^2)
    M22 = Ps[4]*P0.Jbc*cos(th)^2 + Ps[3]*P0.m*(Ps[1]*P0.rc*cos(th)-Ps[2]*P0.h*sin(th))^2 + Ps[5]*P0.Js
    #
    dLdth = -w_psi^2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th)-(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2)) - (Ps[3]*P0.m)*(Ps[6]*P0.g)*((Ps[1]*P0.rc)*cos(th) - (Ps[2]*P0.h)*sin(th))
    #
    dp_psi_dth = -2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th) + (Ps[3]*P0.m)*(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2))*w_psi
    #
    F_th = Ps[7]*P0.K_th_th*U[1] - Ps[8]*P0.K_th_psi*U[2] - P0.D_th*w_th + K[1]
    F_psi = Ps[9]*P0.K_psi_th*U[1] - Ps[10]*P0.K_psi_psi*U[2] - P0.D_psi*w_psi + K[2]

    #
    dx[1] = w_th
    dx[2] = w_psi
    dx[3] = (dLdth + F_th)/M11
    dx[4] = (-dp_psi_dth*w_th + F_psi)/M22
end

uode_f! = (dx,x,p,t) -> helicopter_uode(dx,x,Ps_best,P0,u_t,p,t)
prob = ODEProblem(uode_f!,x0_best,tspan)
sol_best = solve(prob,AutoTsit5(TRBDF2(autodiff=false)),reltol=1e-6)

model_univ_ux = FastChain(FastDense(6, 16, tanh),
                       FastDense(16, 2))
pinit = initial_params(model_univ_ux)/1000

function helicopter_uode(dx,x,Ps,P0,u,p,t)
# P0 = (rc=rc,h=h,m=m,Jbc=Jbc,Js=Js,g=g,K_th_th=K_th_th,K_th_psi=K_th_psi,K_psi_th=K_psi_th,K_psi_psi=K_psi_psi,D_th=D_th,D_psi=D_psi);
    th = x[1]
    psi = x[2]
    w_th = x[3]
    w_psi = x[4]

    U = [u.th(t),u.psi(t)]
    K = model_univ_ux([U;x],p)
    #
    M11 = Ps[4]*P0.Jbc + Ps[3]*P0.m*((Ps[1]*P0.rc)^2+(Ps[2]*P0.h)^2)
    M22 = Ps[4]*P0.Jbc*cos(th)^2 + Ps[3]*P0.m*(Ps[1]*P0.rc*cos(th)-Ps[2]*P0.h*sin(th))^2 + Ps[5]*P0.Js
    #
    dLdth = -w_psi^2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th)-(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2)) - (Ps[3]*P0.m)*(Ps[6]*P0.g)*((Ps[1]*P0.rc)*cos(th) - (Ps[2]*P0.h)*sin(th))
    #
    dp_psi_dth = -2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th) + (Ps[3]*P0.m)*(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2))*w_psi
    #
    F_th = Ps[7]*P0.K_th_th*U[1] - Ps[8]*P0.K_th_psi*U[2] - P0.D_th*w_th + K[1]
    F_psi = Ps[9]*P0.K_psi_th*U[1] - Ps[10]*P0.K_psi_psi*U[2] - P0.D_psi*w_psi + K[2]

    #
    dx[1] = w_th
    dx[2] = w_psi
    dx[3] = (dLdth + F_th)/M11
    dx[4] = (-dp_psi_dth*w_th + F_psi)/M22
end

uode_f! = (dx,x,p,t) -> helicopter_uode(dx,x,Ps_best,P0,u_t,p,t)

tspan = (tm[1],tm[end])



#
prob = ODEProblem(uode_f!,x0_best,tspan,pnn)
#
sol_nn = solve(prob,AutoTsit5(TRBDF2(autodiff=false)),reltol=1e-6)

plot(sol_best,vars=(rad2deg2,0,1),lw=LW1,lc=LC1,label=L"$\theta$")
plot!(sol_best,vars=(rad2deg2,0,1),lw=LW1,lc=LC3,label=L"$\theta$")
plot!(tm,theta*180/pi,lw=LW1,ls=LS2,lc=LC2,label=L"\theta^\mathrm{d}")
plot!(xlim = tspan,xlabel=L"time $t$ [s]", ylabel=L"$\theta$ [$^\circ$]")
plot!(title="Pitch angle: model (blue, solid) vs. data (red, dotted)",box=true)
figname="Helicopter_pitch-angle_model_fit_augmented.svg"
savefig(figpath*figname)

plot(sol_best,vars=(rad2deg2,0,2),lw=LW1,lc=LC1,label=L"$\psi$")
plot!(sol_nn,vars=(rad2deg2,0,2),lw=LW1,lc=LC3,label=L"$\psi$")
plot!(tm,psi*180/pi,lw=LW1,ls=LS2,lc=LC2,label=L"\psi^\mathrm{d}")
plot!(xlim = tspan,xlabel=L"time $t$ [s]", ylabel=L"$\psi$ [$^\circ$]")
plot!(title="Yaw angle: model (blue, solid) vs. data (red, dotted)",box=true)
figname="Helicopter_yaw-angle_model_fit_augmented.svg"
savefig(figpath*figname)

open("../optimization_results/equation_discovery_ux.txt", "w") do f
    println(f,Ψ)
    println(f)
    print_equations(f,Ψ)
    println(f)
    print_equations(f,Ψ,show_parameter=true)
    println(f)
    write(f,"parameters = $(Ψ.parameters) \n")
end
