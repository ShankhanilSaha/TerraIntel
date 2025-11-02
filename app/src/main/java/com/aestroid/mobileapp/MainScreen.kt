package com.aestroid.mobileapp

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.aestroid.mobileapp.ViewModel.MainViewModel
import com.aestroid.mobileapp.UnitConfig

@OptIn(ExperimentalMaterial3Api::class) 
@Composable
fun MainScreen(
    viewModel: MainViewModel = viewModel()
) {
    val context = LocalContext.current
    val locationStatus by viewModel.locationStatus.collectAsState()
    
    
    var unitIdText by remember { mutableStateOf(UnitConfig.getUnitId(context)) }
    var unitTypeText by remember { mutableStateOf(UnitConfig.getUnitType(context)) }
    var showUnitConfig by remember { mutableStateOf(false) }
    
    
    val snackbarHostState = remember { SnackbarHostState() }

    
    var showPermissionDeniedSnackbar by remember { mutableStateOf(false) }
    var showBackgroundPermissionDeniedSnackbar by remember { mutableStateOf(false) }
    var showBackgroundPermissionRationale by remember { mutableStateOf(false) }

    
    val isAndroid10Plus = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q

    
    fun hasForegroundLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED ||
        ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.ACCESS_COARSE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }

    
    fun hasBackgroundLocationPermission(): Boolean {
        return if (isAndroid10Plus) {
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_BACKGROUND_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        } else {
            true 
        }
    }

    
    val foregroundLocationLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestMultiplePermissions(),
        onResult = { permissions ->
            val foregroundGranted = permissions.getOrDefault(
                Manifest.permission.ACCESS_FINE_LOCATION,
                false
            ) || permissions.getOrDefault(
                Manifest.permission.ACCESS_COARSE_LOCATION,
                false
            )

            if (foregroundGranted) {
                
                if (isAndroid10Plus && !hasBackgroundLocationPermission()) {
                    
                    showBackgroundPermissionRationale = true
                } else {
                    
                    startLocationService(context)
                }
            } else {
                
                showPermissionDeniedSnackbar = true
            }
        }
    )

    
    val backgroundLocationLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            
            startLocationService(context)
        } else {
            
            showBackgroundPermissionDeniedSnackbar = true
            
            startLocationService(context)
        }
        showBackgroundPermissionRationale = false
    }

    
    if (showBackgroundPermissionRationale) {
        AlertDialog(
            onDismissRequest = {
                showBackgroundPermissionRationale = false
                
                startLocationService(context)
            },
            title = { Text("Background Location Required") },
            text = {
                Text(
                    "This app needs background location permission to track your location even when the app is in the background. " +
                            "This allows continuous location tracking for military operations."
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showBackgroundPermissionRationale = false
                        backgroundLocationLauncher.launch(
                            Manifest.permission.ACCESS_BACKGROUND_LOCATION
                        )
                    }
                ) {
                    Text("Grant Permission")
                }
            },
            dismissButton = {
                TextButton(
                    onClick = {
                        showBackgroundPermissionRationale = false
                        
                        startLocationService(context)
                    }
                ) {
                    Text("Cancel")
                }
            }
        )
    }

    
    if (showPermissionDeniedSnackbar) {
        androidx.compose.runtime.LaunchedEffect(snackbarHostState) {
            snackbarHostState.showSnackbar(
                message = "Location permission is required for tracking. Please grant permission in Settings.",
                duration = SnackbarDuration.Long
            )
            showPermissionDeniedSnackbar = false
        }
    }
    
    
    if (showBackgroundPermissionDeniedSnackbar) {
        androidx.compose.runtime.LaunchedEffect(snackbarHostState) {
            snackbarHostState.showSnackbar(
                message = "Background location permission denied. Location will only update when app is in foreground.",
                duration = SnackbarDuration.Long
            )
            showBackgroundPermissionDeniedSnackbar = false
        }
    }

    
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Military Location Tracker") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Black,
                    titleContentColor = MaterialTheme.colorScheme.primary
                ),
                actions = {
                    IconButton(onClick = { showUnitConfig = !showUnitConfig }) {
                        Icon(
                            imageVector = Icons.Default.Info,
                            contentDescription = "Unit Configuration"
                        )
                    }
                }
            )
        },
        snackbarHost = { SnackbarHost(hostState = snackbarHostState) }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(16.dp))
            
            
            if (showUnitConfig) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            text = "Unit Configuration",
                            style = MaterialTheme.typography.titleMedium
                        )
                        
                        OutlinedTextField(
                            value = unitIdText,
                            onValueChange = { unitIdText = it },
                            label = { Text("Unit ID") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true
                        )
                        
                        OutlinedTextField(
                            value = unitTypeText,
                            onValueChange = { unitTypeText = it },
                            label = { Text("Unit Type") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                            supportingText = { 
                                Text("Examples: tank, helicopter, soldier, drone, vehicle, mobile")
                            }
                        )
                        
                        Button(
                            onClick = {
                                UnitConfig.setUnitId(context, unitIdText)
                                UnitConfig.setUnitType(context, unitTypeText)
                                showUnitConfig = false
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Save Configuration")
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .defaultMinSize(minHeight = 150.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Column(
                    modifier = Modifier
                        .padding(24.dp)
                        .fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    when (val state = locationStatus) {
                        is MainViewModel.LocationStatus.Idle -> {
                            Icon(
                                imageVector = Icons.Default.LocationOn,
                                contentDescription = "Idle",
                                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.size(48.dp)
                            )
                            Text(
                                text = "Location tracking not started",
                                style = MaterialTheme.typography.bodyLarge,
                                textAlign = TextAlign.Center
                            )
                            Text(
                                text = "Unit: ${UnitConfig.getUnitId(context)} (${UnitConfig.getUnitType(context)})",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        is MainViewModel.LocationStatus.Success -> {
                            Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = "Success",
                                tint = MaterialTheme.colorScheme.primary,
                                modifier = Modifier.size(48.dp)
                            )
                            Text(
                                text = "Location Sent",
                                style = MaterialTheme.typography.titleLarge,
                                textAlign = TextAlign.Center
                            )
                            Text(
                                text = "Unit: ${state.unitId}",
                                style = MaterialTheme.typography.bodyMedium
                            )
                            Text(
                                text = "Coordinates: ${String.format("%.6f", state.lat)}, ${String.format("%.6f", state.lon)}",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        is MainViewModel.LocationStatus.Error -> {
                            Icon(
                                imageVector = Icons.Default.Error,
                                contentDescription = "Error",
                                tint = MaterialTheme.colorScheme.error,
                                modifier = Modifier.size(48.dp)
                            )
                            Text(
                                text = "Error",
                                style = MaterialTheme.typography.titleMedium,
                                color = MaterialTheme.colorScheme.error,
                                textAlign = TextAlign.Center
                            )
                            Text(
                                text = state.message,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.error,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            
            Button(
                onClick = {
                    
                    if (hasForegroundLocationPermission()) {
                        
                        if (isAndroid10Plus && !hasBackgroundLocationPermission()) {
                            
                            showBackgroundPermissionRationale = true
                        } else {
                            
                            startLocationService(context)
                        }
                    } else {
                        
                        foregroundLocationLauncher.launch(
                            arrayOf(
                                Manifest.permission.ACCESS_FINE_LOCATION,
                                Manifest.permission.ACCESS_COARSE_LOCATION
                            )
                        )
                    }
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Start Location Tracking")
            }
            
            Spacer(modifier = Modifier.height(16.dp))
        }
    }
}


private fun startLocationService(context: Context) {
    val intent = Intent(context, LocationService::class.java)
    context.startService(intent)
}