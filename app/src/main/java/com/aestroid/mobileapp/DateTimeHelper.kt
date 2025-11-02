package com.aestroid.mobileapp

import java.text.SimpleDateFormat
import java.util.*

object DateTimeHelper {
    
    fun getCurrentISTTimestamp(): String {
        val istTimeZone = TimeZone.getTimeZone("Asia/Kolkata")
        val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        dateFormat.timeZone = istTimeZone
        return dateFormat.format(Date())
    }
}

