package com.st10371629.coherechatapp.network

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object RetrofitClient {
    val api: CohereApi = Retrofit.Builder()
        .baseUrl("https://cohere-serverr-qegb.onrender.com/") // render URL
        .addConverterFactory(GsonConverterFactory.create())
        .build()
        .create(CohereApi::class.java)
}
