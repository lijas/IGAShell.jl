using Five
using IgAShell

#Dimension
DIM = 3
NELX = 51
NELY = 5
ORDERS = (3,3,3)
@assert(isodd(NELY)) #So outputed cell for stresses is in the center

#Geometry
P = 1.0
L = 10.0
h = 0.1
b = 1.0

#
angles = deg2rad.([0.0, 0.0])

data = ProblemData(
    dim = DIM,
    tend = 1.0,
    adaptive = true
)

interfacematerial = MatCZBilinear(
    K    = 1.0e5,
    Gᴵ   = (0.5, 0.5, 0.5),
    τᴹᵃˣ = (50.0, 50.0, 50.0),
    η    = 1.6
) 

material(_α) = 
MatLinearElastic(
    E = 200e5,
    nu = 0.0
) 

layermats = [material(α) for α in angles]

#
nurbsmesh = IgAShell.IGA.generate_nurbsmesh((NELX, NELY), ORDERS[1:2], (L, b), sdim=DIM) 
data.grid = IgAShell.IGA.convert_to_grid_representation(nurbsmesh)

#Setsk and variables
nlayers = length(angles)
ninterfaces = nlayers - 1
addedgeset!(data.grid, "left", (x)-> x[1] ≈ 0.0)
topfaceset = [FaceIndex(cellid, 2) for cellid in 1:getncells(data.grid)]
addedgeset!(data.grid, "tip", (x)->x[1] ≈ L)
partset1 = collect(1:length(data.grid.cells))
postcell = ceil(Int, 0.5*(NELX*NELY))

#
cellstates = [IgAShell.LAYERED for i in 1:getncells(data.grid)]
interface_damage = zeros(ninterfaces, getncells(data.grid))

#IGAshell data
igashelldata = 
IgAShell.IGAShellData(;
    layer_materials           = layermats,
    interface_material        = interfacematerial,
    viscocity_parameter       = 0.0,
    orders                    = ORDERS,
    knot_vectors              = nurbsmesh.knot_vectors,
    thickness                 = h,
    width                     = DIM == 2 ? b : 1.0,
    initial_cellstates        = cellstates,
    initial_interface_damages = interface_damage,
    nqp_inplane_order         = 3,
    nqp_ooplane_per_layer     = 5,
    adaptable                 = false,
    small_deformations_theory = true,
    LIMIT_UPGRADE_INTERFACE   = 0.04,
    nqp_interface_order       = 4
)  

igashell = 
IgAShell.IGAShell(
    cellset = partset1, 
    connectivity = reverse(nurbsmesh.IEN, dims=1), 
    data = igashelldata
) 
push!(data.parts, igashell)

#
data.output[] = Output(
    interval = 0.0,
    runname = "cantilever_beam",
    savepath = "./"
)

#
etf = IGAShellWeakBC( 
    set = getedgeset(data.grid, "left"),
    func = (x,t) -> zeros(DIM), 
    comps = 1:DIM,
    igashell = igashell, 
    penalty = 1e12
)
push!(data.constraints, etf)

#Force
etf = IGAShellExternalForce(
    set = topfaceset, 
    func = (x,t) -> Vec{DIM}((0.0, 0.0, t* -P/(b*L))),
    igashell = igashell
)
push!(data.external_forces, etf)

#
output = OutputData(
    type = IgAShell.IGAShellBCOutput(
        igashell = igashell,
        comps = [DIM]
    ),
    interval = 0.0,
    set      = getedgeset(data.grid, "tip")
)
data.outputdata["maxdisp"] = output

#
vtkoutput = VTKCellOutput(
    type = IGAShellConfigStateOutput()
)
Five.push_vtkoutput!(data.output[], vtkoutput)

#
vtkoutput = VTKNodeOutput(
    type = MaterialStateOutput(
        field = :interface_damage
    ),
)
Five.push_vtkoutput!(data.output[], vtkoutput)

output = OutputData(
    type = IGAShellStressOutput(
        igashell = igashell,
    ),
    interval = 0.0,
    set      = [postcell]
)
data.outputdata["Stress at 50%"] = output

#
output = OutputData(
    type = IGAShellRecovoredStressOutput(
        igashell = igashell,
    ),
    interval = 0.0,
    set      = [postcell]
)
data.outputdata["RS at 50%"] = output

#
state, globaldata = build_problem(data) do dh, parts, dbc
    instructions = IgAShell.initial_upgrade_of_dofhandler(dh, igashell)
    Five.update_dofhandler!(dh, StateVariables(Float64, ndofs(dh)), instructions)
    
    alldofs = collect(1:ndofs(dh))
    JuAFEM.copy!!(dbc.free_dofs, alldofs)
end

solver = NewtonSolver(
    Δt0 = 1.0, 
    Δt_max = 1.0, 
    tol=1e-5
)


output = solvethis(solver, state, globaldata)

if true
    using Plots; pyplot(); PyPlot.pygui(true)

    d = [output.outputdata["maxdisp"].data[i].displacements for i in 1:length(output.outputdata["maxdisp"].data)]
    f = [output.outputdata["maxdisp"].data[i].forces for i in 1:length(output.outputdata["maxdisp"].data)]

    σᶻˣ_vec = getproperty.(output.outputdata["RS at 50%"].data[end][1], :σᶻˣ)
    σᶻʸ_vec = getproperty.(output.outputdata["RS at 50%"].data[end][1], :σᶻʸ)
    σᶻᶻ_vec = getproperty.(output.outputdata["RS at 50%"].data[end][1], :σᶻᶻ)
    ζ_vec = getproperty.(output.outputdata["RS at 50%"].data[end][1], :ζ)

    fig = plot(reuse=false, layout = (2,3))
    plot!(fig[4], σᶻᶻ_vec, ζ_vec, label = "recovered")
    plot!(fig[5], σᶻʸ_vec, ζ_vec, label = "recovered")
    plot!(fig[6], σᶻˣ_vec, ζ_vec, label = "recovered")

    #
    σxx_driver = getindex.(output.outputdata["Stress at 50%"].data[end][1].stresses, 1, 1)
    σyy_driver = getindex.(output.outputdata["Stress at 50%"].data[end][1].stresses, 2, 2)
    σxy_driver = getindex.(output.outputdata["Stress at 50%"].data[end][1].stresses, 1, 2)

    zcoords = getindex.(output.outputdata["Stress at 50%"].data[end][1].local_coords, 3)
    σᶻˣ_driver = getindex.(output.outputdata["Stress at 50%"].data[end][1].stresses, 1, 3)
    σᶻʸ_driver = getindex.(output.outputdata["Stress at 50%"].data[end][1].stresses, 2, 3)
    σᶻᶻ_driver = getindex.(output.outputdata["Stress at 50%"].data[end][1].stresses, 3, 3)
    zcoords = getindex.(output.outputdata["Stress at 50%"].data[end][1].local_coords, 3)
    zcoords .-= mean(zcoords)

    plot!(fig[1], σxx_driver, zcoords, label = "driver")
    plot!(fig[2], σyy_driver, zcoords, label = "driver")
    plot!(fig[3], σxy_driver, zcoords, label = "driver")
    plot!(fig[4], σᶻᶻ_driver, zcoords, label = "driver")
    plot!(fig[5], σᶻʸ_driver, zcoords, label = "driver")
    plot!(fig[6], σᶻˣ_driver, zcoords, label = "driver")
end



