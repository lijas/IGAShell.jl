#
#
#
struct IGAShellExternalForce{P<:IGAShell} <: Five.AbstractExternalForce
    faceset::GeometryObjectVectors
    traction::Function
    igashell::Base.RefValue{P}
end

function IGAShellExternalForce(cellset, traction::Function, igashell::IGAShell)

    return IGAShellExternalForce{typeof(igashell)}(collect(cellset), traction, Base.RefValue(igashell))
end

function Five.apply_external_force!(dh::JuAFEM.AbstractDofHandler, ef::IGAShellExternalForce{P}, state::StateVariables, prev_state::StateVariables, system_arrays::SystemArrays{T}, globaldata) where {P,T}
    
    #Igashell extract
    igashell = ef.igashell[]
    dim_s = JuAFEM.getdim(igashell)

    #Coords
    ncoords = JuAFEM.nnodes_per_cell(igashell)
    X = zeros(Vec{dim_s,T}, ncoords)
    Xᵇ = zeros(Vec{dim_s,T}, ncoords)

    #Traction force
    traction = ef.traction
    A = 0.0
    for faceidx in ef.faceset
        #Note: faceidx can be FaceIndex EdgeIndex, CellIndex(Int) etc...
        cellid = faceidx[1]

        #Celldofs
        ndofs = ndofs_per_cell(dh, cellid)
        celldofs = zeros(Int, ndofs)
        fe = zeros(T, ndofs)
        ke = zeros(T, ndofs, ndofs)

        #Get coords and dofs of cell
        JuAFEM.cellcoords!(X, dh, cellid)
        JuAFEM.celldofs!(celldofs, dh, cellid)

        #Bezier and cell values
        local_cellid = findfirst((i)->i==cellid, igashell.cellset)
        Ce = get_extraction_operator(intdata(igashell), local_cellid)
        Xᵇ .= IGA.compute_bezier_points(Ce, X)
        
        #
        @timeit "build" cv = build_facevalue!(igashell, faceidx)
        IGA.set_bezier_operator!(cv, Ce)
        reinit!(cv, Xᵇ)
        
        @timeit "integrate" A += _compute_igashell_external_traction_force!(cv, Xᵇ, traction, faceidx, state.t, fe, getwidth(layerdata(igashell)))
       
        system_arrays.fᵉ[celldofs] += fe
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