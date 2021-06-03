export IGAShellWeakConstraint
#
#
#

struct IGAShellWeakConstraint{P<:IGAShell} <: Five.AbstractExternalForce
    faceset::GeometryObjectVectors
    prescribed_displacement::Function
    igashell::Base.RefValue{P}
    penalty::Float64
end

function IGAShellWeakConstraint(; 
    set, 
    func::Function,
    igashell::IGAShell{dim_p,dim_s,T},
    penalty::T
    ) where {dim_p,dim_s,T}
    
    @assert( 1 == length(func(zero(Vec{dim_s,T}), 0.0)) )

    return IGAShellWeakConstraint{typeof(igashell)}(collect(set), func, Base.RefValue(igashell), penalty)
end

function Five.init_external_force!(a::IGAShellWeakConstraint, ::Ferrite.AbstractDofHandler)
    return a
end

function Five.apply_external_force!(wb::IGAShellWeakConstraint, state::StateVariables{T}, globaldata) where {T}
    
    
    #Igashell extract
    dh = globaldata.dh
    igashell = wb.igashell[]
    dim_s = Ferrite.getdim(igashell)

    #Coords
    ncoords = Ferrite.nnodes_per_cell(igashell)
    X = zeros(Vec{dim_s,T}, ncoords)
    Xᵇ = zeros(Vec{dim_s,T}, ncoords)
    
    for (ic, faceidx) in enumerate(wb.faceset)
        cellid, faceid = faceidx

        ndofs = ndofs_per_cell(dh, cellid)
        
        celldofs = zeros(Int, ndofs)
        fe = zeros(T, ndofs)
        ke = zeros(T, ndofs, ndofs)

        local_cellid = findfirst((i)->i==cellid, igashell.cellset)
        #
        Ce = get_extraction_operator(intdata(igashell), local_cellid)
        
        #Get coords and dofs of cell
        Ferrite.cellcoords!(X, dh, cellid)
        Ferrite.celldofs!(celldofs, dh, cellid)
        Xᵇ .= IGA.compute_bezier_points(Ce, X)

        ue = state.d[celldofs]
        
        #
        @timeit "build" cv = build_facevalue!(igashell, faceidx)

        IGA.set_bezier_operator!(cv, Ce)
        reinit!(cv, Xᵇ)

        @timeit "integrate" _compute_igashell_weak_constraint_condition!(cv, Xᵇ, wb.prescribed_displacement, faceidx, state.t, ke, fe, ue, wb.penalty, getwidth(layerdata(igashell)))
        
        state.system_arrays.fᵉ[celldofs] += fe
        state.system_arrays.Kᵉ[celldofs, celldofs] += ke
    end

end

function _compute_igashell_weak_constraint_condition!(fv::IGAShellValues, coords::AbstractVector{Vec{dim,T}}, prescr_disp::Function, faceidx, time::T, ke::AbstractMatrix, fe::AbstractVector, ue::AbstractVector, stiffness::T, width::T) where {dim,T}
    
    for q_point in 1:getnquadpoints(fv)
        dΓ = getdetJdA(fv, q_point, faceidx) * width
        u = function_value(fv, q_point, ue)
        
        X = spatial_coordinate(fv, q_point, coords)
        
        dg, g = Tensors.gradient(u -> prescr_disp(u,time), u, :all)

        for i in 1:getnbasefunctions(fv)
            δui = basis_value(fv, q_point, i)  
            
            fe[i] += stiffness*g*dg ⋅ δui * dΓ

            for j in 1:getnbasefunctions(fv)   
                δuj = basis_value(fv, q_point, j)

                ke[i,j] += stiffness*  δuj ⋅ (dg ⊗ dg) ⋅ δui  * dΓ
            end
        end
    end

end
