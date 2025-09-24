package com.st10371629.coherechatapp.network

import okhttp3.ResponseBody
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.POST

interface CohereApi {
    @POST("/query")
    fun query(@Body body: Map<String, String>): Call<ResponseBody>
}
