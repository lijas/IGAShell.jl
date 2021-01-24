using Five
using IgAShell
#using JLD2; using FileIO

function build_cantilever_beam_example()

    T = Float64
    DIM = 3
    nelx = 50
    nely = 1
    nelz = 1
    nqp_cz = 4
    bc_penalty = 1e12

    #Geometry
    P = 1.0
    L = 10.0
    h = 0.1
    b = 1.0
    angles = deg2rad.(T[0.0, 90.0, 0.0, 90.0])
    nlayers = length(angles)

    #Definition for 2d
    addvertexset! = JuAFEM.addvertexset!
    getvertexset = JuAFEM.getvertexset
    VertexInterfaceIndex = IgAShell.VertexInterfaceIndex
    nels = (nelx,)
    beamsize = (L,)

    if DIM == 3
        addvertexset! = JuAFEM.addedgeset!
        getvertexset = JuAFEM.getedgeset
        VertexInterfaceIndex = IgAShell.EdgeInterfaceIndex
        nels = (nelx, nely)
        beamsize = (L, b)
    end

    #
    data = ProblemData(
        dim = DIM,
        tend = 1.0,
    )

    interfacematerial = MatCZBilinear(
        K    = 10.0e6,
        Gᴵ   = (0.5, 0.5, 0.5),
        τᴹᵃˣ = 1000 .* (50.0, 50.0, 50.0),
        η    = 1.6
    )

    material = MatLinearElastic(E = 200e5, nu = 0.0)
    layermats = [Material2D(material, Five.PLANE_STRESS) for α in angles]
    layermats = [material for α in angles]

    #Mesh
    orders = ntuple(x->2, DIM-1)
    r = 2 #order in z

    nurbsmesh = IgAShell.IGA.generate_nurbsmesh(nels, orders, beamsize, sdim=DIM)
    data.grid = IgAShell.IGA.convert_to_grid_representation(nurbsmesh)

    #
    cellstates = [IgAShell.LAYERED for i in 1:prod(nels)]
    interface_damage = zeros(T, nlayers-1, getncells(data.grid))
    
    #Sets
    partset1 = collect(1:length(data.grid.cells))
    addvertexset!(data.grid, "left",    (x)-> x[1] ≈ 0.0)
    addvertexset!(data.grid, "right",   (x)-> x[1] ≈ L)
    faceset = [FaceIndex(cellid, 2) for cellid in 1:getncells(data.grid)]

    #IGAshell data
    igashelldata = 
    IgAShell.IGAShellData(;
        layer_materials           = layermats,
        interface_material        = interfacematerial,
        viscocity_parameter       = 0.0,
        orders                    = (orders..., r),
        knot_vectors              = nurbsmesh.knot_vectors,
        thickness                 = h,
        initial_cellstates        = cellstates,
        nqp_inplane_order         = 3,
        nqp_ooplane_per_layer     = 3,
        width                     = DIM == 2 ? b : 1.0,
        adaptable                 = false,
        small_deformations_theory = false,
        nqp_interface_order       = 3
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
        runname = "cantilever",
        savepath = "./"
    )

    #BC
    etf = IGAShellWeakBC( 
        set = getvertexset(data.grid, "left"),
        func = (x,t) -> zeros(T, DIM), 
        comps = 1:DIM,
        igashell = igashell, 
        penalty = bc_penalty
    )
    push!(data.constraints, etf)

    #Force
    forcevec = Vec(ntuple(i-> i == DIM ? -1.0/(b*L) : 0.0, DIM))
    etf = IGAShellExternalForce(
        set = faceset, 
        func = (x,t) -> forcevec,
        igashell = igashell
    )
    push!(data.external_forces, etf)

    #Output
    output = OutputData(
        type = IgAShell.IGAShellBCOutput(
            igashell = igashell,
            comps = [DIM]
        ),
        interval = 0.0,
        set      = getvertexset(data.grid, "right")
    )
    data.outputdata["reactionforce"] = output

    #Post cell 1
    #=
    @assert(isodd(nels[dim-1]))
    postcell = ceil(Int, 0.5*prod(nels))

    stress_output = eliasfem.IGAShellStressOutput(Ref(igashell), cellset = [postcell], interval = 0.01)
    data.outputs["Stress at 50%"] = stress_output
    stress_output = eliasfem.IGAShellRecovoredStressOutput(Ref(igashell), cellset = [postcell], interval = 0.01)
    data.outputs["RS at 50%"] = stress_output
    stress_output = eliasfem.IGAShellIntegrationValuesOutput(Ref(igashell), cellset = [postcell], interval = 0.01)
    data.outputs["intvalues"] = stress_output
    
    addvertexset!(grid, "tip", (x)->x[1] ≈ L)
    edgeset = getvertexset(grid, "tip")
    data.outputs["maxdisp"] = eliasfem.IGAShellBCOutput(Ref(igashell), outputset = edgeset, components = [dim], interval = 0.00)
    =#

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
        tol=1e-3,
        maxitr_first_step = 60,
    )

    return solver, state, globaldata
end

solver, state, globaldata = build_cantilever_beam_example()
output = solvethis(solver, state, globaldata)

1==1