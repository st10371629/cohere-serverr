package com.st10371629.coherechatapp

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInAccount
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.android.gms.cast.framework.SessionManager
import com.google.android.gms.common.api.ApiException
import com.st10371629.coherechatapp.auth.BiometricHelper
import com.st10371629.coherechatapp.auth.SessionManager
import com.st10371629.coherechatapp.network.RetrofitClient
import okhttp3.ResponseBody
import org.json.JSONObject
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response

class MainActivity : ComponentActivity() {

    private lateinit var sessionManager: SessionManager
    private lateinit var biometricHelper: BiometricHelper

    // Google Sign-In result launcher
    private val signInLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val data = result.data
        if (data != null) {
            val task = GoogleSignIn.getSignedInAccountFromIntent(data)
            try {
                val account = task.getResult(ApiException::class.java)
                handleSignInAccount(account)
            } catch (e: ApiException) {
                Log.e("Auth", "Google sign in failed code=${e.statusCode}", e)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        sessionManager = SessionManager(this)
        biometricHelper = BiometricHelper(this)

        setContent {
            CohereChatAppTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ChatAppScreen(
                        sessionManager = sessionManager,
                        biometricHelper = biometricHelper,
                        startGoogleSignIn = { startGoogleSignIn() }
                    )
                }
            }
        }
    }

    private fun startGoogleSignIn() {
        val gso = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
            .requestEmail()
            .requestIdToken(getString(R.string.server_client_id)) // from Google Cloud Console
            .build()
        val client = GoogleSignIn.getClient(this, gso)
        val signInIntent = client.signInIntent
        signInLauncher.launch(signInIntent)
    }

    private fun handleSignInAccount(account: GoogleSignInAccount?) {
        account ?: return
        val idToken = account.idToken
        val email = account.email ?: account.displayName ?: "unknown"
        if (!idToken.isNullOrBlank()) {
            sessionManager.saveSession(idToken, email)
            Log.d("Auth", "Saved session for $email")
        } else {
            Log.w("Auth", "No idToken received. Check server_client_id in strings.xml")
        }
    }
}
@Composable
fun ChatAppScreen(
    sessionManager: SessionManager,
    biometricHelper: BiometricHelper,
    startGoogleSignIn: () -> Unit
) {
    var userInput by remember { mutableStateOf("") }
    var botReply by remember { mutableStateOf("Bot response will appear here.") }
    var signedIn by remember { mutableStateOf(false) }
    var userEmail by remember { mutableStateOf<String?>(null) }
    var hasStoredSession by remember { mutableStateOf(false) }

    // Check stored session at startup
    LaunchedEffect(Unit) {
        val token = sessionManager.getIdToken()
        val email = sessionManager.getEmail()
        hasStoredSession = token != null && email != null
        if (hasStoredSession && biometricHelper.canAuthenticate()) {
            biometricHelper.showBiometricPrompt(
                title = "Unlock chat",
                onSuccess = {
                    signedIn = true
                    userEmail = email
                },
                onFailure = { reason -> Log.d("Biometric", "Biometric fail: $reason") }
            )
        }
    }

    Scaffold { inner ->
        Column(
            modifier = Modifier
                .padding(inner)
                .padding(16.dp)
        ) {
            Text("Cohere Chat", style = MaterialTheme.typography.headlineSmall)
            Spacer(Modifier.height(12.dp))

            if (!signedIn) {
                Text("You are not signed in.")
                Spacer(Modifier.height(8.dp))

                Button(onClick = { startGoogleSignIn() }, modifier = Modifier.fillMaxWidth()) {
                    Text("Sign in with Google")
                }

                Spacer(Modifier.height(8.dp))

                if (hasStoredSession && biometricHelper.canAuthenticate()) {
                    Button(
                        onClick = {
                            biometricHelper.showBiometricPrompt(
                                title = "Unlock chat",
                                onSuccess = {
                                    signedIn = true
                                    userEmail = sessionManager.getEmail()
                                },
                                onFailure = { reason -> Log.d("Biometric", "Auth error: $reason") }
                            )
                        },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Unlock with Biometric")
                    }
                }

                Spacer(Modifier.height(16.dp))
            } else {
                Text("Signed in as: ${userEmail ?: "Unknown"}", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(12.dp))

                Button(
                    onClick = {
                        sessionManager.clearSession()
                        GoogleSignIn.getClient(LocalContext.current, GoogleSignInOptions.DEFAULT_SIGN_IN).signOut()
                        signedIn = false
                        userEmail = null
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Sign out")
                }
                Spacer(Modifier.height(16.dp))
            }

            // Chat input + send
            OutlinedTextField(
                value = userInput,
                onValueChange = { userInput = it },
                label = { Text("Enter your message") },
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(Modifier.height(8.dp))

            Button(
                onClick = {
                    val requestBody = mapOf("message" to userInput)
                    RetrofitClient.api.query(requestBody).enqueue(object : Callback<ResponseBody> {
                        override fun onResponse(call: Call<ResponseBody>, response: Response<ResponseBody>) {
                            if (response.isSuccessful) {
                                val responseString = response.body()?.string()
                                val json = JSONObject(responseString ?: "{}")
                                botReply = json.optString("reply", "No reply key in response")
                            } else {
                                botReply = "Server error: ${response.code()}"
                            }
                        }

                        override fun onFailure(call: Call<ResponseBody>, t: Throwable) {
                            botReply = "Request failed: ${t.localizedMessage}"
                        }
                    })
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Send")
            }

            Spacer(Modifier.height(24.dp))

            Text("Bot says:", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            Text(botReply)
        }
    }
}
