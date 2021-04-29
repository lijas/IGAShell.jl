export IGAShellExternalForce
#
#
#
struct IGAShellExternalForce{P<:IGAShell} <: Five.AbstractExternalForce
    faceset::GeometryObjectVectors
    traction::Function
    igashell::Base.RefValue{P}
end

function IGAShellExternalForce(; 
    set, 
    func::Function, 
    igashell::IGAShell{dim_p, dim_s, T}) where {T, dim_p, dim_s}

    @assert( dim_s == length(func(zero(Vec{dim_s,T}), 0.0)) )


    return IGAShellExternalForce{typeof(igashell)}(collect(set), func, Base.RefValue(igashell))
end

function Five.init_external_force!(a::IGAShellExternalForce, ::Ferrite.AbstractDofHandler)
    return a
end

function Five.apply_external_force!(ef::IGAShellExternalForce{P}, state::StateVariables{T}, globaldata) where {P,T}
    
    #Igashell extract
    dh = globaldata.dh
    igashell = ef.igashell[]
    dim_s = Ferrite.getdim(igashell)

    #Coords
    ncoords = Ferrite.nnodes_per_cell(igashell)
    X = zeros(Vec{dim_s,T}, ncoords)
    Xᵇ = zeros(Vec{dim_s,T}, ncoords)

    #Traction force
    traction = ef.traction
    A = 0.0
    for faceidx in ef.faceset
        #Note: faceidx can be FaceIndex EdgeIndex, CellIndex(Int) etc...
        cellid = faceidx[1]
        local_cellid = findfirst((i)->i==cellid, igashell.cellset)
        cellstate = getcellstate(igashell, local_cellid)

        #Celldofs
        ndofs = ndofs_per_cell(dh, cellid)
        celldofs = zeros(Int, ndofs)
        fe = zeros(T, ndofs)
        ke = zeros(T, ndofs, ndofs)

        #Get coords and dofs of cell
        Ferrite.cellcoords!(X, dh, cellid)
        Ferrite.celldofs!(celldofs, dh, cellid)

        #Bezier and cell values
        Ce = get_extraction_operator(intdata(igashell), local_cellid)
        Xᵇ .= IGA.compute_bezier_points(Ce, X)
        
        #
        @timeit "build" cv = build_facevalue!(igashell, faceidx)
        IGA.set_bezier_operator!(cv, Ce)
        reinit!(cv, Xᵇ)
        
        @timeit "integrate" A += _compute_igashell_external_traction_force!(cv, Xᵇ, traction, faceidx, state.t, fe, getwidth(layerdata(igashell)))
       
        state.system_arrays.fᵉ[celldofs] += fe
    end

end

function _compute_igashell_external_traction_force!(fv::IGAShellValues, Xᵇ, traction::Function, faceidx, time::Float64, fe::AbstractVector, width)

    A = 0.0
    for q_point in 1:getnquadpoints(fv)
        dΓ = getdetJdA(fv, q_point, faceidx) * width
        X = spatial_coordinate(fv, q_point, Xᵇ)

        t = traction(X,time)
        A += dΓ
        for i in 1:getnbasefunctions(fv)
            δui = basis_value(fv, q_point, i)
            fe[i] += (δui ⋅ t) * dΓ
        end
    end

    return A
end