package com.aestroid.mobileapp

import android.app.Notification
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import com.google.android.gms.location.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import com.aestroid.mobileapp.UnitConfig

class LocationService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    
    private lateinit var fusedLocationClient: FusedLocationProviderClient

    
    private lateinit var locationCallback: LocationCallback

    override fun onCreate() {
        super.onCreate()
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        
        locationCallback = object : LocationCallback() {
            override fun onLocationResult(locationResult: LocationResult) {
                super.onLocationResult(locationResult)

                locationResult.lastLocation?.let { location ->
                    val lat = location.latitude
                    val lon = location.longitude
                    Log.d("LocationService", "New location: $lat, $lon")

                    
                    
                    serviceScope.launch {
                        val unitId = UnitConfig.getUnitId(this@LocationService)
                        val unitType = UnitConfig.getUnitType(this@LocationService)
                        DataRepository.sendLocation(unitId, unitType, lat, lon)
                    }
                }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        

        
        val notification = createNotification()

        
        
        startForeground(1, notification)

        
        startLocationUpdates()

        
        return START_STICKY
    }

    private fun startLocationUpdates() {
        
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {

            Log.e("LocationService", "Location permission not granted. Stopping service.")
            stopSelf() 
            return
        }

        
        val locationRequest = LocationRequest.create().apply {
            interval = 10000 
            fastestInterval = 5000 
            priority = Priority.PRIORITY_HIGH_ACCURACY
        }

        
        fusedLocationClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            Looper.getMainLooper() 
        )
    }

    private fun createNotification(): Notification {
        
        return NotificationCompat.Builder(this, "location") 
            .setContentTitle("Location Tracking Active")
            .setContentText("Your location is being sent to the server.")
            
            
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true) 
            .build()
    }

    override fun onDestroy() {
        super.onDestroy()
        
        fusedLocationClient.removeLocationUpdates(locationCallback)
        
        serviceScope.cancel()
    }

    
    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
}