
# Try to find OpenVDB project dll/so and headers
#


# outputs
unset(OPENVDB_DLL CACHE)
unset(OPENVDB_LIB CACHE)
unset(OPENVDB_FOUND CACHE)
unset(OPENVDB_INCLUDE_DIR CACHE)
unset(OPENVDB_LIB_DIR CACHE)
unset(OPENVDB_BIN_DIR CACHE)
unset(OPENVDB_LIB_DEBUG CACHE)
unset(OPENVDB_LIB_RELEASE CACHE)

set(VCPKG_BIN_DIR "${VCPKG_ROOT_DIR}/installed/x64-windows/bin" CACHE PATH "path")
set(VCPKG_BIN_DEBUG_DIR "${VCPKG_ROOT_DIR}/installed/x64-windows/debug/bin" CACHE PATH "path")
set(VCPKG_LIB_DIR "${VCPKG_ROOT_DIR}/installed/x64-windows/lib" CACHE PATH "path")
set(VCPKG_LIB_DEBUG_DIR "${VCPKG_ROOT_DIR}/installed/x64-windows/debug/lib" CACHE PATH "path")
set(VCPKG_INCLUDE_DIR "${VCPKG_ROOT_DIR}/installed/x64-windows/include" CACHE PATH "path")

FUNCTION(package_openvdb_binaries)
   list ( APPEND OPENVDB_LIST "blosc.dll")
   list ( APPEND OPENVDB_LIST "boost_system-vc141-mt-x64-1_69.dll")
   list ( APPEND OPENVDB_LIST "boost_thread-vc141-mt-x64-1_69.dll")
   list ( APPEND OPENVDB_LIST "Half-2_3.dll")
   list ( APPEND OPENVDB_LIST "Iex-2_3.dll")
   list ( APPEND OPENVDB_LIST "IexMath-2_3.dll")
   list ( APPEND OPENVDB_LIST "IlmImf-2_3.dll")
   list ( APPEND OPENVDB_LIST "IlmThread-2_3.dll")
   list ( APPEND OPENVDB_LIST "Imath-2_3.dll")
   list ( APPEND OPENVDB_LIST "openvdb.dll")
   list ( APPEND OPENVDB_LIST "tbb.dll")
   list ( APPEND OPENVDB_LIST "zlib1.dll")
   set ( OPENVDB_BINARIES ${OPENVDB_LIST} PARENT_SCOPE)      
ENDFUNCTION()


if(USE_OPENVDB) 
	set ( OPENVDB_LIB_DIR "${VCPKG_LIB_DIR}" CACHE PATH "path" )
	set ( OPENVDB_BIN_DIR "${VCPKG_BIN_DIR}" CACHE PATH "path" )
	set ( OPENVDB_INCLUDE_DIR "${VCPKG_INCLUDE_DIR}" CACHE PATH "path" )
	if(WIN32)
		_find_files( OPENVDB_DLL VCPKG_BIN_DIR "blosc.dll" )
		_find_files( OPENVDB_DLL VCPKG_BIN_DIR "Half-2_3.dll" )    
		_find_files( OPENVDB_DLL VCPKG_BIN_DIR "Iex-2_3.dll" )    
		_find_files( OPENVDB_DLL VCPKG_BIN_DIR "IexMath-2_3.dll" )    
		_find_files( OPENVDB_DLL VCPKG_BIN_DIR "IlmThread-2_3.dll" )    
		_find_files( OPENVDB_DLL VCPKG_BIN_DIR "Imath-2_3.dll" )    	
		_find_files( OPENVDB_DLL VCPKG_BIN_DIR "tbb.dll" )  
		
		# -------- Locate LIBS
		_find_files( OPENVDB_LIB_RELEASE VCPKG_LIB_DIR "openvdb.lib" )	
	
	
	endif(WIN32)
	
	_find_files( OPENVDB_HEADERS VCPKG_INCLUDE_DIR "Openvdb/openvdb.h" )
	
	if (OPENVDB_DLL)
	  set( OPENVDB_FOUND "YES" )      
	  
	endif()
	
	
endif()


include(FindPackageHandleStandardArgs)

SET(OPENVDB_DLL ${OPENVDB_DLL} CACHE PATH "path")
SET(OPENVDB_LIB_DEBUG ${OPENVDB_LIB_DEBUG} CACHE PATH "path")
SET(OPENVDB_LIB_RELEASE ${OPENVDB_LIB_RELEASE} CACHE PATH "path")

mark_as_advanced( OPENVDB_FOUND )

